import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import sys
import random
from _datetime import datetime

import data
import model
import musicVAE


# PARAMETERS FROM GOOGLE PAPER:
batch_size = 512
learning_rate = 0.001      # gets annealed to 0.00001
#exp_decay_rate = 0.9999
beta = 0.2                 # gets annealed from 0 to 0.2 for 2 bars, but is 0.2 for 16 bars


def training(num_epochs, batch_size, bars, pianoroll, transpose, verbose, save_location, resume=False, initialize=False):

    def loss_func(x_hat, x, mean, std_deviation, beta):
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        bce = bce / (batch_size * bars)
        kl = -0.5 * torch.sum(1 + torch.log(std_deviation ** 2) - mean ** 2 - std_deviation ** 2)
        kl = kl / batch_size
        loss = bce + beta * kl

        if verbose:
            print("\t\tCross Entropy: \t{}\n\t\tKL-Divergence: \t{}\n\t\tFinal Loss: \t{}".format(bce, kl, loss))

        return loss

    # no teacher forcing is used during evaluation, the model has to rely on its own previous outputs
    def evaluate():
        vae.eval()
        avg_loss = 0
        i = 0

        for batch in eval_loader:
            i += 1
            if verbose:
                print("\tbatch " + str(i) + ":")
            batch = batch.to(device)
            vae_output, mu, log_var = vae(batch)
            loss = criterion(vae_output, batch, mu, log_var, beta)
            avg_loss += loss.item()
        avg_loss = avg_loss / (len(eval_set) / batch_size)
        vae.train()
        return avg_loss

    if save_location[0] != '/' and save_location[0] != '~':
        save_location = data.path_to_root + '/' + save_location

    stride = 1 if transpose else 3
    train_set = data.FinalDataset('train', bars, stride=stride, pianoroll=pianoroll, transpose=transpose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_set = data.FinalDataset('validation', bars, stride=stride, pianoroll=pianoroll, transpose=transpose)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size)

    loss_list = []
    resume_epoch = 0
    min_eval_loss = sys.maxsize     # does not remember min loss if trainig is aborted and resumed, but all losses are saved in loss_list()

    if initialize:
        vae = musicVAE.get_pretrained_vae(bars=bars, pianoroll=pianoroll, transpose=transpose, verbose=verbose)
        if verbose:
            print("loaded the pretrained weights")
    else:
        vae = model.VAE(bars, pianoroll)
    vae.train()

    if resume:
        checkpoint = musicVAE.get_checkpoint(str(train_set), save_location)
        prefix = "last_"
        loss_list = checkpoint[prefix + 'loss_list']
        resume_epoch = checkpoint[prefix + 'epoch']

        print("found checkpoint\n\tresuming training at epoch " + str(resume_epoch) + ("\n\tlast train loss: {}\n\tlast eval loss: {}\n".format(loss_list[-1][0], loss_list[-1][1]) if loss_list else "\n"))

        vae.load_state_dict(checkpoint[prefix + 'vae_state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    print("using device " + str(device) + "\n")

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
    if resume:
        optimizer.load_state_dict(checkpoint[prefix + 'optim_state_dict'])
    criterion = loss_func

    #compute initial evaluation loss before training
    if not resume:
        if verbose:
            print("Initial Evaluation\n\n")
        initial_loss = evaluate()
        print("\nInitial evaluation loss: " + str(initial_loss))

    for epoch in range(resume_epoch, num_epochs):
        if verbose:
            print("\n\n\n\nSTARTING EPOCH " + str(epoch) + "\n")
        avg_loss = 0
        avg_correct = 0
        i = 0

        for batch in train_loader:
            i += 1
            batch = batch.to(device)        # .to() returns a copy for tensors!

            optimizer.zero_grad()

            if verbose:
                print("\tbatch " + str(i) + ":")
            vae.set_ground_truth(batch)
            vae_output, mu, log_var = vae(batch)
            loss = criterion(vae_output, batch, mu, log_var, beta)

            # to free some memory:
            vae_output = None
            batch = None

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

        if verbose:
            print("\n\n\nEvaluation of epoch " + str(epoch) + "\n")

        avg_loss = avg_loss / (len(train_set) / batch_size)
        eval_loss = evaluate()
        timestamp = datetime.now()
        timestamp = "{}.{}.  -  {}:{}".format(timestamp.day, timestamp.month, timestamp.hour, timestamp.minute)


        print("EPOCH " + str(epoch) + "\t\t(finished at:   " + timestamp + ")")
        print("\ttraining loss: \t\t" + str(avg_loss))
        loss_list.append((avg_loss, eval_loss))
        print("\tevaluation loss: \t" + str(eval_loss) + "\n")
        best = False
        if eval_loss <= min_eval_loss:
            min_eval_loss = eval_loss
            best = True

        if verbose:
            print(("\n\tsaving checkpoint.."))
        musicVAE.save(vae, optimizer, loss_list, epoch+1, str(train_set), best, save_location)


# requires pianoroll inputs, convert before calling this function
def compute_correct_notes(seq_1, seq_2, use_monophonic_for_comparison):

    if seq_1.shape != seq_2.shape:
        raise ValueError("Input mismatch: the input sequences don't have the same shape")

    if use_monophonic_for_comparison:
        seq_1 = data.small_to_full_pianoroll(seq_1)
        seq_1 = data.pianoroll_to_monophonic_repr(seq_1)
        seq_2 = data.small_to_full_pianoroll(seq_2)
        seq_2 = data.pianoroll_to_monophonic_repr(seq_2)

    correct_notes = 0
    for slice_1, slice_2 in zip(seq_1, seq_2):
        if slice_1.max() == 0 and slice_2.max() == 0:       # could also be omitted if not using MIDI note zero as input, because then slice.argmax() == 0 is not ambiguous
            correct_notes += 1
        elif slice_1.argmax() == slice_2.argmax():
            correct_notes += 1

    return correct_notes


def evaluate_correct_notes(bars, comparisons=8000, use_monophonic_for_comparison=False, pianoroll=True, transpose=True, verbose=False):
    vae = musicVAE.get_trained_vae(bars, pianoroll, transpose, verbose)
    dset = data.FinalDataset('validation', bars, stride=1, pianoroll=pianoroll, transpose=transpose)

    if verbose:
        print("Number of correct notes in the reconstruction of each clip:")

    total_notes = 0
    total_correct = 0
    best = 0
    worst = 1

    it = random.sample(range(0, len(dset)), comparisons)
    n=0
    for i in it:
        snippet = dset.__getitem__(i)
        snippet = snippet.unsqueeze(0)

        n += 1
        if n in range(0, comparisons, 500) or n == 100:
            print(n, '\t\t{}/{}  =  {} correct'.format(total_correct, total_notes, total_correct/total_notes))

        reconstructed, _, _ = vae(snippet)
        reconstructed = data.model_output_to_pianoroll(reconstructed, pianoroll)
        snippet = snippet.squeeze()
        if not pianoroll:
            snippet = data.monophonic_repr_to_pianoroll(snippet)
        else:
            snippet = snippet[:, 0:88]     # remove pause tokens

        length = snippet.shape[0]
        correct_notes = compute_correct_notes(snippet, reconstructed, use_monophonic_for_comparison)
        total_notes += length
        total_correct += correct_notes

        ratio = correct_notes / length
        if ratio > best:
            best = ratio
        if ratio < worst:       # dont use an elif, because possibly the first value will be the worst
            worst = ratio

        if verbose:
            print("\t{}/{} = {}% correct".format(correct_notes, length, ratio*100))

    print("best accuracy:\t\t{}%".format(best*100))
    print("worst accuracy:\t\t{}%".format(worst*100))
    print("average accuracy:\t\t{}%".format((total_correct / total_notes) * 100))
