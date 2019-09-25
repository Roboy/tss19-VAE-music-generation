import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import data

lstm_1_hidden_size = 2048
lstm_1_layers = 2
latent_dimension = 512

lstm_conductor_hidden_size = 1024
lstm_conductor_input_size = 1     # conductor gets only zeros as inputs anyway, so just set this very small.

lstm_l2_decoder_hidden_size = 1024

class VAE(nn.Module):
    def __init__(self, bars, pianoroll=False):
        super(VAE, self).__init__()

        if self.training:
            self.counter = 0
            self.scheduled_sampling_ratio = 0
            self.ground_truth = None

        self.batch_size = None
        self.seq_len = bars * 4 * data.resolution_per_beat
        self.u = bars       # amount of subsequences that conductor layer creates
        self.input_size = 127 if pianoroll else 90

        # encoder
        self.lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=lstm_1_hidden_size, num_layers=lstm_1_layers, bidirectional=True, batch_first=True)
        self.fc_mean = nn.Linear(in_features=lstm_1_hidden_size*2, out_features=latent_dimension)
        self.fc_std_deviation = nn.Linear(in_features=lstm_1_hidden_size*2, out_features=latent_dimension)

        # decoder
            #conductor

        self.fc_2 = nn.Linear(in_features=latent_dimension, out_features=lstm_conductor_hidden_size*4)      # output is used to initialize h and c for both layers of the conductor lstm
        self.lstm_conductor = nn.LSTM(input_size=lstm_conductor_input_size, hidden_size=lstm_conductor_hidden_size, num_layers=2, batch_first=True)

            # second level decoder

        self.fc_3 = nn.Linear(in_features=lstm_conductor_hidden_size, out_features=lstm_l2_decoder_hidden_size*4)   # output is used to initialize h and c for both layers of the l2 lstm
        self.lstm_l2_decoder_cell_1 = nn.LSTMCell(input_size=lstm_conductor_hidden_size+self.input_size, hidden_size=lstm_l2_decoder_hidden_size)
        self.lstm_l2_decoder_cell_2 = nn.LSTMCell(input_size=lstm_l2_decoder_hidden_size, hidden_size=lstm_l2_decoder_hidden_size)
        self.fc_4 = nn.Linear(in_features=lstm_l2_decoder_hidden_size, out_features=self.input_size)


    def encode(self, t):

        # input of shape (batch, seq_len, input_size)

        # hidden of shape  (num_layers * num_directions, batch, hidden_size) only if batch_first == False!!

        _, (h, _) = self.lstm_1(t)       # TODO input t, (initial hidden state, initial cell state) for better training?

        h_t = h.view(lstm_1_layers, 2, self.batch_size, lstm_1_hidden_size)    # 2 = num_directions
        h_t_forward = h_t[1, 0, :, :]
        h_t_backward = h_t[1, 1, :, :]
        h_t = torch.cat((h_t_forward, h_t_backward), dim=1)

        z_mean = self.fc_mean(h_t)

        z_std_deviation = self.fc_std_deviation(h_t)
        z_std_deviation = torch.log1p(torch.exp(z_std_deviation))

        return z_mean, z_std_deviation

    def reparameterize(self, mean, std_deviation):
        return mean + torch.randn_like(mean) * std_deviation

    def l2_decode(self, embedding, previous):

        t = self.fc_3(embedding)
        t = torch.tanh(t)

        h1 = t[:, 0:lstm_l2_decoder_hidden_size]
        h2 = t[:, lstm_l2_decoder_hidden_size:2 * lstm_l2_decoder_hidden_size]
        c1 = t[:, 2 * lstm_l2_decoder_hidden_size:3 * lstm_l2_decoder_hidden_size]
        c2 = t[:, 3 * lstm_l2_decoder_hidden_size:4 * lstm_l2_decoder_hidden_size]

        outputs = []

        for _ in range(self.seq_len//self.u):

            if self.training:
                if self.counter > 0 and random.random() > self.scheduled_sampling_ratio:
                    previous = self.ground_truth[self.counter - 1]
                    # TODO catch exception and print understable error message ("The provided ground truth did not have the correct dimensions" or if is None "Model is in training mode but no ground truth was provided for the forward pass")
                else:
                    previous = previous.detach()        # needed?

            l2_in = torch.cat((embedding, previous), dim=1)
            h1, c1 = self.lstm_l2_decoder_cell_1(l2_in, (h1, c1))
            h2, c2 = self.lstm_l2_decoder_cell_2(h1, (h2, c2))
            previous = self.fc_4(h2)
            outputs.append(previous)

        return outputs

    def decode(self, z):

        # get initial states for conductor lstm

        t = self.fc_2(z)
        t = torch.tanh(t)

        h1 = t[None, :, 0:lstm_conductor_hidden_size]
        h2 = t[None, :, lstm_conductor_hidden_size:2 * lstm_conductor_hidden_size]
        c1 = t[None, :, 2 * lstm_conductor_hidden_size:3 * lstm_conductor_hidden_size]
        c2 = t[None, :, 3 * lstm_conductor_hidden_size:4 * lstm_conductor_hidden_size]

        h = torch.cat((h1, h2), dim=0)
        c = torch.cat((c1, c2), dim=0)

        # get embeddings from conductor

        conductor_input = torch.zeros(size=(self.batch_size, self.u, lstm_conductor_input_size))

        embeddings, _ = self.lstm_conductor(conductor_input, (h, c))
        # embeddings = embeddings.permute(1, 0, 2)
        embeddings = torch.unbind(embeddings, dim=1)

        # decode embeddings

        outputs = []
        previous = torch.zeros((self.batch_size, self.input_size))

        for emb in embeddings:
            l2_out = self.l2_decode(emb, previous)
            outputs.extend(l2_out)
            previous = l2_out[-1]

        output_tensor = torch.stack(outputs, dim=1)

        output_tensor = output_tensor.softmax(dim=2)

        return output_tensor


    def forward(self, t):
        self.batch_size = t.shape[0]
        z_mean, z_std_deviation = self.encode(t)
        z = self.reparameterize(z_mean, z_std_deviation)
        out = self.decode(z)
        if self.training:
            self.ground_truth = None        # not necessary, but ensures that the old ground truth cant accidentally be reused in the next step
            self.counter = 0
        return out, z_mean, z_std_deviation

    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth

    def set_scheduled_sampling_ratio(self, ratio):      # ratio is the probability with which the model uses its own previous output instead of teacher forcing
        self.scheduled_sampling_ratio = ratio
