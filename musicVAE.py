#!/usr/bin/env python

"""MusicVAE

usage:
  musicVAE.py train <epochs> <batch> <bars> [-p | --pianoroll] [-d=<path> | --dataset-location=<path>]
  musicVAE.py generate-dataset <split> <bars> <stride> [-p | --pianoroll] [-d=<path> | --dataset-location=<path>] [-m=<path2> | --maestro-location=<path2>]
  musicVAE.py (-h | --help)

options:
  -h --help                             Show this screen.
  -p --pianoroll                        Use a standard pianoroll representation for the data instead of the default monophonic one.
  -d=<path> --dataset-location=<path>   Enter a File Path to the .h5 file containing the dataset(s) [default: Data/FinalDataset/final_dataset.h5]
  -m=<path2> --maestro-location=<path2>]Enter a File Path to the folder containing the Maestro Dataset [default: "/media/micaltu/opensuse/home/micalt/PycharmProjects/Test2/Data/maestro-v2.0.0"

commands:
   train                                Train the network.
   generate-dataset                     Generate a dataset that can be used for training or testing/validation.

"""

# from docopt import docopt

TRAIN = """usage: musicVAE.py train <epochs> <batch> <bars> [-p | --pianoroll] [-d=<path> | --dataset-location=<path>]

  -h --help                             Show this screen.
  -p --pianoroll                        Use a standard pianoroll representation for the data instead of the default monophonic one.
  -d=<path> --dataset-location=<path>   Enter a File Path to the .h5 file containing the dataset(s) [default: Data/FinalDataset/final_dataset.h5]
"""

GENERATEDATASET = """usage: musicVAE.py generate-dataset <split> <bars> <stride> [-p | --pianoroll] [-d=<path> | --dataset-location=<path>] [-m=<path2> | --maestro-location=<path2>]

  -h --help                             Show this screen.
  -p --pianoroll                        Use a standard pianoroll representation for the data instead of the default monophonic one.
  -d=<path> --dataset-location=<path>   Enter a File Path to the .h5 file containing the dataset(s) [default: Data/FinalDataset/final_dataset.h5]
  -m=<path2> --maestro-location=<path2> Enter a File Path to the folder containing the Maestro Dataset [default: "/media/micaltu/opensuse/home/micalt/PycharmProjects/Test2/Data/maestro-v2.0.0"
"""


import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import argparse
# from schema import Schema, And, Use, Optional
import matplotlib.pyplot as plt
import sys
import model
import data


# PARAMETERS FROM GOOGLE PAPER:
batch_size = 512
learning_rate = 0.001      # gets annealed to 0.00001
#exp_decay_rate = 0.9999
beta = 0.2                 # gets annealed for 2 bars, but is 0.25 for 16 bars

verbose = False


def _parse_train(args):
    # num_epochs = args.epochs
    # batch_size = args.batch
    # bars = args.bars
    if args.verbose:
        verbose = True
    print("starting training with:\n\t{} epochs\n\t{} batch size\n\t{} bars".format(args.epochs, args.batch, args.bars))
    repr = "standard"
    if args.pianoroll:
        repr = "pianoroll"
    print("\tusing " + repr + " representation")
    print("\tverbose is set to " + str(args.verbose))
    if args.resume:
        print("\ttrying to resume training from previous checkpoint\n")

    training(args.epochs, args.batch, args.bars, args.pianoroll, args.verbose, args.save_location, args.resume)

def _parse_show_datasets(args):
    print("the following datasets are available at the moment:\n")
    dsets = data.get_available_datasets(args.size)
    if args.size:
        i = iter(dsets)
        for d, s in zip(i, i):
            print('\t' + d + '\t\t\twith size ' + str(s))
    else:
        for d in dsets:
            print('\t' + d)
    print("\nto generate additional datasets see musicVAE generate-dataset --help")


def _parse_generate_dataset(args):
    data.create_final_dataset(args.split, args.bars, args.stride, args.pianoroll)

def main():

    # arguments = docopt(__doc__, options_first=True)
    # if arguments['<command>'] == 'train':
    #     num_epochs = arguments['<epochs>']
    #     batch_size = arguments['<batch-size>']
    #     bars = arguments['<bars>']
    #     print("starting training with:\n\t{} epochs\n\t{} batch size\n\t{} bars\n".format(num_epochs, batch_size, bars))
    #     training()
    # elif arguments['<command>'] == 'generate-dataset':
    #     data.create_final_dataset(arguments['split'], arguments['bars'], arguments['stride'])
    # else:
    #     exit("{0} is not a command. \
    #           See 'musicVAE.py --help'.".format(arguments['<command>']))

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # train parser

    train_parser = subparsers.add_parser('train')

    train_parser.add_argument(
        "epochs",
        type=int,
        help="number of epochs to train for"
    )

    train_parser.add_argument(
        "batch",
        type=int,
        help="size of one batch"
    )

    train_parser.add_argument(
        "bars",
        type=int,
        help='length of in- and output for the model in bars'
    )

    train_parser.add_argument(
        '-d', '--dataset_location',
        help='path to the .h5 file storing the dataset',
        required=False,
        default='Data/FinalDataset/final_dataset.h5'
    )

    train_parser.add_argument(
        '-s', '--save_location',
        help='directory to store the training checkpoints in',
        required=False,
        default='Models/Checkpoints'
    )

    train_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a pianoroll representation as in- and output'
    )

    train_parser.add_argument(
        '-r', '--resume',
        action='store_true',
        help='resume the training from a previously saved checkpoint'
    )

    train_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='get more information during the training process'
    )

    train_parser.set_defaults(func=_parse_train)



    # show datasets parser

    show_datasets_parser = subparsers.add_parser('show-datasets')

    show_datasets_parser.add_argument(
        '-s', '--size',
        action='store_true',
        help='also show the size of the datasets'
    )

    show_datasets_parser.set_defaults(func=_parse_show_datasets)



    # generate-dataset parser

    generate_dataset_parser = subparsers.add_parser('generate-dataset')

    generate_dataset_parser.add_argument(
        "split",
        help="can be one of the following: train, test or validation"
    )

    generate_dataset_parser.add_argument(
        "bars",
        type=int,
        help='length of the sequences in the resulting dataset in bars'
    )

    generate_dataset_parser.add_argument(
        "stride",
        type=int,
        help="stride in bars (how big are the steps with which a full performance from the Maestro Dataset gets stepped over)"
    )

    generate_dataset_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a pianoroll representation instead of the default monophonic representation'
    )

    # generate_dataset_parser.add_argument(
    #     '-d', '--dataset_location',
    #     help='path to the .h5 file storing the dataset',
    #     required=False,
    #     default='Data/FinalDataset/final_dataset.h5'
    # )
    #
    # generate_dataset_parser.add_argument(
    #     '-m', '--maestro_location',
    #     help='path to the root folder of the Maestro Dataset',
    #     required=False,
    #     default='Data/FinalDataset/final_dataset.h5'
    # )

    generate_dataset_parser.set_defaults(func=_parse_generate_dataset)


    args = parser.parse_args()
    args.func(args)


def save(model, optimizer, loss_list, epoch, dset_name, best, location):
    path = location + '/' + dset_name + ".tar"
    prefix_list = ["best_", "last_"] if best else ["last_"]
    for prefix in prefix_list:
        torch.save({
            prefix + 'vae_state_dict': model.state_dict(),
            prefix + 'optim_state_dict': optimizer.state_dict(),
            prefix + 'loss_list': loss_list,
            prefix + 'epoch': epoch
        }, path)


def get_checkpoint(dset_name, location):
    # maybe this could also be loaded directly to the GPU if available
    path = location + '/' + dset_name + ".tar"
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except:
         raise FileNotFoundError("Could not find a previosuly saved checkpoint to resume training from.")
    return checkpoint



def training(num_epochs, batch_size, bars, pianoroll, verbose, save_location, resume=False):

    def loss_func(x_hat, x, mean, std_deviation, beta):
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        bce = bce / (batch_size * bars)
        kl = -0.5 * torch.sum(1 + torch.log(std_deviation ** 2) - mean ** 2 - std_deviation ** 2)
        kl = kl / batch_size
        loss = bce + beta * kl

        if verbose:
            print("\t\tCross Entropy: \t{}\n\t\tKL-Divergence: \t{}\n\t\tFinal Loss: \t{}".format(bce, kl, loss))

        return loss

    def evaluate():
        # TODO currently vae does not use teacher forcing during evaluation, try both options?
        vae.eval()
        total_loss = 0
        i = 0

        for batch in eval_loader:
            i += 1
            if verbose:
                print("\tbatch " + str(i) + ":")
            batch = batch.to(device)
            vae_output, mu, log_var = vae(batch)
            loss = criterion(vae_output, batch, mu, log_var, beta)
            total_loss += loss.item()

        vae.train()
        return total_loss


    train_set = data.FinalDataset('train', bars, stride=1, pianoroll=pianoroll)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    eval_set = data.FinalDataset('validation', bars, stride=1, pianoroll=pianoroll)
    eval_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size)

    loss_list = []
    resume_epoch = 0
    min_eval_loss = sys.maxsize     # does not remember min loss if trainig is aborted and resumed, but all losses are saved in loss_list()

    vae = model.VAE(bars, pianoroll)
    vae.train()

    if resume:
        checkpoint = get_checkpoint(str(train_set), save_location)
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

    for epoch in range(resume_epoch, num_epochs):
        if verbose:
            print("\n\n\n\nSTARTING EPOCH " + str(epoch) + "\n")
        total_loss = 0
        total_correct = 0
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

            total_loss += loss.item()

        if verbose:
            print("\n\n\nEvaluation of epoch " + str(epoch) + "\n")

        eval_loss = evaluate()
        print("EPOCH " + str(epoch))
        print("\ttraining loss: \t\t" + str(total_loss))
        loss_list.append((total_loss, eval_loss))
        print("\tevaluation loss: \t" + str(eval_loss) + "\n")
        best = False
        if eval_loss <= min_eval_loss:
            min_eval_loss = eval_loss
            best = True

        if verbose:
            print(("\n\tsaving checkpoint.."))
        save(vae, optimizer, loss_list, epoch+1, str(train_set), best, save_location)



if __name__ == "__main__":
    main()

