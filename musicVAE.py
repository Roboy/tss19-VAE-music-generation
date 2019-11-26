#!/usr/bin/env python

import torch
import argparse
import random
import model
import data
import training


def _parse_train(args):
    print("starting training with:\n\t{} epochs\n\t{} batch size\n\t{} bars".format(args.epochs, args.batch, args.bars))
    repr = "standard MIDI-like"
    if args.pianoroll:
        repr = "pianoroll"
    print("\tusing " + repr + " representation")
    if args.transpose:
        print('\tusing traning data transposed to only one major and one minor key')
    else:
        print('\tusing training data transposed to every possible key')
    print("\tverbose is set to " + str(args.verbose))
    if args.resume:
        if args.initialize:
            raise ValueError("Resuming training from previous checkpoint and using pretrained weights is not possible!")
        print("\ttrying to resume training from previous checkpoint")
    if args.initialize:
        print("\ttrying to load pretrained weights from 2 bar model\n")

    training.training(args.epochs, args.batch, args.bars, args.pianoroll, args.transpose, args.verbose, args.save_location, args.resume, args.initialize)

def _parse_show_datasets(args):
    print("the following datasets are available at the moment:\n")
    dsets = data.get_available_datasets(args.size)
    if args.size:
        i = iter(dsets)
        for d, s in zip(i, i):
            spaces = 70
            spaces -= len(d)
            if spaces <=0:
                spaces = 10
            print('\t' + d + '.' * spaces + 'with size ' + str(s))
    else:
        for d in dsets:
            print('\t' + d)
    print("\nto generate additional datasets see musicVAE generate-dataset --help")


def _parse_generate_dataset(args):
    data.create_final_dataset(args.split, args.bars, args.stride, args.pianoroll, args.transpose)

def _parse_delete_dataset(args):
    data.delete_dataset(args.split, args.bars, args.stride, args.pianoroll, args.transpose)

def _parse_sample(args):
    vae = get_trained_vae(args.bars, args.pianoroll, args.transpose, args.verbose)
    sample = vae.sample()
    data.pianoroll_to_midi(sample, args.save_location)

def _parse_interpolate(args):
    vae = get_trained_vae(args.bars, args.pianoroll, args.transpose, args.verbose)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.start_sequence:
        path = args.start_sequence
        if path[0] != '/' and path[0] != '~':
            path = data.path_to_root + '/' + path
        midi = data.midi_to_small_one_hot_pianoroll(path)
        if args.verbose:
            print("read the start sequence " + path)
        if(midi.shape[0] != args.bars * 16):
            print("The start sequence does not have the correct length as specified by the bars parameter!")
        midi = midi.unsqueeze(dim=0)
        midi.to(device)
        m, v = vae.encode(midi)
        z = vae.reparameterize(m, v)
    else:
        z = torch.randn((1, model.latent_dimension), requires_grad=False, device=device)
        if args.verbose:
            print("generated a random start sequence")

    if args.end_sequence:
        path = args.end_sequence
        if path[0] != '/' and path[0] != '~':
            path = data.path_to_root + '/' + path
        midi = data.midi_to_small_one_hot_pianoroll(path)
        if args.verbose:
            print("read the end sequence " + path)
        if (midi.shape[0] != args.bars * 16):
            print("The end sequence does not have the correct length as specified by the bars parameter!")
        midi = midi.unsqueeze(dim=0)
        midi.to(device)
        m, v = vae.encode(midi)
        z_end = vae.reparameterize(m, v)
    else:
        z_end = torch.randn((1, model.latent_dimension), requires_grad=False, device=device)
        if args.verbose:
            print("generated a random end sequence")

    step_vector = (z_end - z) / (args.steps + 1)
    step_vector.to(device)

    for i in range(args.steps+2):
        sample = vae.sample(z)
        filename = args.save_location + "/interpolate_" + str(i) + ".midi"
        data.pianoroll_to_midi(sample, filename)
        z = z + step_vector

#TODO add gpu support to reconstruct
def _parse_reconstruct(args):
    vae = get_trained_vae(args.bars, args.pianoroll, args.transpose, args.verbose)

    # read start sequence
    if args.start_sequence:
        path = args.start_sequence
        if path[0] != '/' and path[0] != '~':
            path = data.path_to_root + '/' + path
        seq = data.midi_to_small_one_hot_pianoroll(path)
        if(seq.shape[0] != args.bars * 16):
            print("The start sequence does not have the correct length as specified by the bars parameter!")

    # ..or generate a random one
    else:
        if args.verbose:
            print("no start sequence given, a random one is generated")
        ds = data.FinalDataset('train', bars=args.bars, pianoroll=args.pianoroll)
        i = random.randint(0, len(ds))
        seq = ds.__getitem__(i)

    # save start sequence
    start_seq = data.monophonic_repr_to_pianoroll(seq)
    filename = args.save_location + "/reconstruct_before.midi"
    data.pianoroll_to_midi(start_seq, filename)

    seq = seq.unsqueeze(dim=0)

    # reconstruct the sequence
    for _ in range(args.number_reconstructions):
        seq, _, _ = vae(seq)

    # save end sequence
    seq = data.model_output_to_pianoroll(seq, args.pianoroll)
    filename = args.save_location + "/reconstruct_after.midi"
    data.pianoroll_to_midi(seq, filename)



def get_trained_vae(bars, pianoroll=False, transpose=False, verbose=False):
    vae = model.VAE(bars, pianoroll)
    vae.eval()

    location = data.path_to_root + "Models/Final"
    stride = 1 if transpose else 3
    dset_name = data.get_dataset_name("train", bars, stride, pianoroll, transpose)

    checkpoint = get_checkpoint(dset_name, location)
    if verbose:
        print("loaded Checkpoint " + dset_name)

    prefix = "last_"
    vae.load_state_dict(checkpoint[prefix + 'vae_state_dict'])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    if verbose:
        print("using device " + str(device) + "\n")

    return vae

def get_pretrained_vae(bars, pianoroll=False, transpose=False, verbose=False):
    vae = model.VAE(bars, pianoroll)
    vae.train()

    location = data.path_to_root + "Models/Final"
    stride = 1 if transpose else 3
    dset_name = data.get_dataset_name("train", bars, stride, pianoroll, transpose)
    if pianoroll:
        dset_name = "pianoroll_" + dset_name

    checkpoint = get_checkpoint(dset_name, location)
    if verbose:
        print("loaded Checkpoint")

    prefix = "last_"
    pretrained_dict = checkpoint[prefix + 'vae_state_dict']

    pretrained_l2 = {k:v for k, v in pretrained_dict.items() if (k[0:7] == "lstm_l2" or k[0:4] == "fc_4")}

    vae.load_state_dict(pretrained_l2, strict=False)
    return vae

def main():

    parser = argparse.ArgumentParser(description="For an explanation of the parameters of any specific command <command>, see its own help:\npython musicVAE.py <command> -h")
    subparsers = parser.add_subparsers()

    # train parser

    train_parser = subparsers.add_parser('train', help="train the model with your choice of dataset or continue the training from a saved checkpoint")

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
        '-t', '--transpose',
        action='store_true',
        help='use the dataset where each sequence is transposed to either C Major or a minor'
    )

    train_parser.add_argument(
        '-r', '--resume',
        action='store_true',
        help='resume the training from a previously saved checkpoint'
    )

    train_parser.add_argument(
        '-i', '--initialize',
        action='store_true',
        help='initialize the parameters of the level 2 LSTM with the pretrained weights from the 2 bar model'
    )

    train_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='get more information during the training process'
    )

    train_parser.set_defaults(func=_parse_train)



    # show datasets parser

    show_datasets_parser = subparsers.add_parser('show-datasets', help="shows all the datasets that are currently available in the HDF file containing all the training data")

    show_datasets_parser.add_argument(
        '-s', '--size',
        action='store_true',
        help='also show the size of the datasets'
    )

    show_datasets_parser.set_defaults(func=_parse_show_datasets)



    # generate-dataset parser

    generate_dataset_parser = subparsers.add_parser('generate-dataset', help="generate a new dataset and save it to the HDF file containing all the training data")

    generate_dataset_parser.add_argument(
        "split",
        help="can be one of the following: train, test or validation; for the training, a train and validation set is required"
    )

    generate_dataset_parser.add_argument(
        "bars",
        type=int,
        help='length of the sequences in the resulting dataset in bars'
    )

    generate_dataset_parser.add_argument(
        "stride",
        type=int,
        help="stride in bars (how big are the steps with which a full performance from the Maestro Dataset gets stepped over; this MusicVAE implementation requires a stride of 1 if the transpose flag is used, and 3 otherwise)"
    )

    generate_dataset_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a pianoroll representation instead of the default MIDI-like representation'
    )

    generate_dataset_parser.add_argument(
        '-t', '--transpose',
        action='store_true',
        help='transpose every performance to either C Major or a minor so that all generated sequences share the same notes'
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



    # delete-dataset parser

    delete_dataset_parser = subparsers.add_parser('delete-dataset', help="delete one of the datasets from the HDF file")

    delete_dataset_parser.add_argument(
        "split",
        help="can be one of the following: train, test or validation"
    )

    delete_dataset_parser.add_argument(
        "bars",
        type=int,
        help='length of the sequences in the resulting dataset in bars'
    )

    delete_dataset_parser.add_argument(
        "stride",
        type=int,
        help="stride in bars (how big are the steps with which a full performance from the Maestro Dataset gets stepped over)"
    )

    delete_dataset_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a pianoroll representation instead of the default MIDI-like representation'
    )

    delete_dataset_parser.add_argument(
        '-t', '--transpose',
        action='store_true',
        help='use the dataset where each sequence is transposed to either C Major or a minor'
    )

    # delete_dataset_parser.add_argument(
    #     '-d', '--dataset_location',
    #     help='path to the .h5 file storing the dataset',
    #     required=False,
    #     default='Data/FinalDataset/final_dataset.h5'
    # )

    delete_dataset_parser.set_defaults(func=_parse_delete_dataset)



    # sample parser

    sample_parser = subparsers.add_parser('sample', help="randomly generate a music sequence and save it as a MIDI file")

    sample_parser.add_argument(
        'bars',
        type=int,
        help='length of the generated sequences in bars'
    )

    sample_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a model trained with sequences in pianoroll representation instead of the default MIDI-like one'
    )

    sample_parser.add_argument(
        '-t', '--transpose',
        action='store_true',
        help='use the model trained with sequences transposed to either C Major or a minor'
    )

    sample_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='get more information during the sampling process'
    )

    sample_parser.add_argument(
        '-s', '--save_location',
        help='path to where the MIDI file will be saved (path has to include filename, default is "Sampled/sample.midi"',
        required=False,
        default='Sampled/sample.midi'
    )

    sample_parser.set_defaults(func=_parse_sample)

    # reconstruct parser

    recon_parser = subparsers.add_parser('reconstruct',
                                          help="encode a sequence and decode it again, one or multiple times")

    recon_parser.add_argument(
        'bars',
        type=int,
        help='length of the generated sequences in bars'
    )

    recon_parser.add_argument(
        '-f', '--start_sequence',
        help='path (absolute or relative without leading /) to the MIDI file containing the sequence for the reconstruction; if no path is given, a random sequence is generated and saved as Sampled/reconstruction_start.midi',
        required=False,
        default=None
    )

    recon_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a model trained with sequences in pianoroll representation instead of the default MIDI-like one'
    )

    recon_parser.add_argument(
        '-t', '--transpose',
        action='store_true',
        help='use the model trained with sequences transposed to either C Major or a minor'
    )

    recon_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='get more information during the reconstruction process'
    )

    recon_parser.add_argument(
        '-s', '--save_location',
        help='directory to store the initial and reconstructed sequence in. Default directory is Sampled',
        required=False,
        default='Sampled'
    )

    recon_parser.add_argument(
        '-n', '--number_reconstructions',
        type=int,
        help='how often the sequence is encoded and decoded, default is 1',
        required=False,
        default=1
    )

    recon_parser.set_defaults(func=_parse_reconstruct)



    # interpolate parser

    interpolate_parser = subparsers.add_parser('interpolate', help="interpolate between two sequences; every generated sequence is saved as a MIDI file")

    interpolate_parser.add_argument(
        "bars",
        type=int,
        help='length of the generated sequences in bars'
    )

    interpolate_parser.add_argument(
        "steps",
        type=int,
        help="the number of intermediate sequences that are generated"
    )

    interpolate_parser.add_argument(
        '-s', '--save_location',
        help='directory to store the interpolations in. Default directory is Sampled',
        required=False,
        default='Sampled'
    )

    interpolate_parser.add_argument(
        '-f', '--start_sequence',
        help='path (absolute or relative without leading /) to the MIDI file containing the start sequence for the interpolation; if no path is given, a random sequence is generated',
        required=False,
        default=None
    )

    interpolate_parser.add_argument(
        '-e', '--end_sequence',
        help='path (absolute or relative without leading /) to the MIDI file containing the end sequence for the interpolation; if no path is given, a random sequence is generated',
        required=False,
        default=None
    )

    interpolate_parser.add_argument(
        '-p', '--pianoroll',
        action='store_true',
        help='use a model trained with sequences in pianoroll representation instead of the default MIDI-like one'
    )

    interpolate_parser.add_argument(
        '-t', '--transpose',
        action='store_true',
        help='use the model trained with sequences transposed to either C Major or a minor'
    )

    interpolate_parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='get more information during the interpolation process'
    )

    interpolate_parser.set_defaults(func=_parse_interpolate)


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
         raise FileNotFoundError("Could not find a previosuly saved checkpoint at " + path)
    return checkpoint



if __name__ == "__main__":
    main()
