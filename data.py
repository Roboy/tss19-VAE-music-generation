import numpy as np
import torch
import pypianoroll as ppr
import pandas
from torch.utils.data.dataset import Dataset
import pretty_midi
import music21
import os
import sys
import h5py


finalDataset_path = '/home/micaltu/tss19-VAE-music-generation/Data/FinalDataset/final_dataset.h5'

# maestro_csv_path = "/home/micalt/PycharmProjects/Test2/Data/maestro-v2.0.0/maestro-v2.0.0.csv"
# maestro_root_dir = "/home/micalt/PycharmProjects/Test2/Data/maestro-v2.0.0"

maestro_root_dir = "/home/micaltu/tss19-VAE-music-generation/Data/maestro-v2.0.0"
maestro_csv_path = maestro_root_dir + "/maestro-v2.0.0.csv"


calculate_correct_tempo = True
false_tempo = 120           # the tempo assigned to every one of the midi files


# Parameters for generating data from performances

resolution_per_beat = 4     # quantized to sixteenth notes if set to 4 (there are 4 sixteenth notes per beat)
# length = model.bars

def get_available_datasets(show_size):
    out = []
    with h5py.File(finalDataset_path, 'a') as f:
        for k in f.keys():
            out.append(str(k))
            if show_size:
                ds = f[k]
                size = ds.shape[0]
                out.append(size)

    return out


def delete_dataset(split, bars, stride, pianoroll):
    dset_name = get_dataset_name(split, bars, stride, pianoroll)
    with h5py.File(finalDataset_path, 'a') as f:
        if dset_name in f.keys():
            del f[dset_name]
            print("deleted dataset " + dset_name)
        else:
            print("dataset " + dset_name + " does not exist")

'''
about Maestro Dataset:

Tempo seems to always be 120 and downbeat in ppr Multitrack object is only True at index 0, rest is false

A piano has (usually, but this is the case for the one used to generate the dataset in the Yamaha e-Competition) 88 keys which correspond to MIDI pitches [21, 108] (including)

'''

class MaestroDataset(Dataset):

    def __init__(self, split='train', bars=2, root_dir=maestro_root_dir):

        if(split != 'train' and split != 'test' and split != 'validation'):
            raise ValueError("class MaestroDataset was initialized with invalid aruments, split has to be 'train', 'test' or 'validation'!")

        self.root_dir = root_dir
        self.bars = bars

        # check whether csv for this split already exists:
        csv_file = maestro_csv_path
        edited_csv_path = maestro_root_dir + "/maestro-v2.0.0_" + split + ".csv"
        if os.path.isfile(edited_csv_path):
            csv_file = edited_csv_path

        self.csv_data_frame = pandas.read_csv(csv_file)

        # if not, generate it
        if csv_file == maestro_csv_path:
            self.csv_data_frame = self.csv_data_frame.drop(labels=['canonical_composer', 'audio_filename'], axis=1)
            #TODO maybe drop more if not all information is needed

            self.csv_data_frame = self.csv_data_frame[self.csv_data_frame['split'] == split]
            self.csv_data_frame.set_axis(range(len(self.csv_data_frame)), axis='index', inplace=True)

            self.compute_tempo()
            self.csv_data_frame.to_csv(path_or_buf=edited_csv_path, index=False, na_rep="NaN")



    def __len__(self):
        return len(self.csv_data_frame)

    def __getitem__(self, index):
        row = self.csv_data_frame.iloc[[index]]

        if calculate_correct_tempo:
            real_tempo = row['tempo']
            real_tempo = float(real_tempo)

            beat_resolution = real_tempo / false_tempo * resolution_per_beat
            beat_resolution = int(round(beat_resolution, 0))

        else:
            beat_resolution = resolution_per_beat

        full_performance = self.get_full_performance(index, beat_resolution)
        return full_performance

    def get_path(self, index):
        row = self.csv_data_frame.iloc[[index]]
        out = row['midi_filename']
        out = out[index]
        return out


    def get_full_performance(self, index, beat_resolution=resolution_per_beat):
        infos = self.csv_data_frame.iloc[index]
        midi_filename = infos['midi_filename']
        path = "" + self.root_dir + "/" + midi_filename

        midi = ppr.parse(filepath=path, beat_resolution=beat_resolution)  # get Multitrack object
        midi = midi.tracks[0]                                # get first/only track
        # midi = midi.pianoroll        has to return track in order for ppr.transpose() to work later
        return midi

    def compute_tempo(self):
        print("computing the real tempos of the performances...")
        tempos = []
        for _, row in self.csv_data_frame.iterrows():       #iterrows returns index and row as pandas Series object
            path = "" + self.root_dir + "/" + row['midi_filename']
            midi_data = pretty_midi.PrettyMIDI(path)
            tem = midi_data.estimate_tempo()
            tempos.append(tem)

        self.csv_data_frame = self.csv_data_frame.assign(tempo=pandas.Series(tempos))

class FinalDataset(Dataset):
    def __init__(self, split='train', bars=2, stride=1, pianoroll=False):
        if (split != 'train' and split != 'test' and split != 'validation'):
            raise ValueError("class FinalDataset was initialized with invalid aruments, split has to be 'train', 'test' or 'validation'!")
        self.split = split
        self.bars = bars
        self.stride = stride
        self.dset = None
        self.dset_name = get_dataset_name(split, bars, stride, pianoroll)

        f = h5py.File(finalDataset_path, 'r')        # TODO does this need to be closed somehow? With doesnt work because f needs to stay open as long as dset is used
        if self.dset_name not in f.keys():
            command = "musicVAE.py generate-dataset " + split + " " + str(bars) + " " + str(stride) + (" --pianoroll" if pianoroll else "")
            raise ValueError("The dataset " + self.dset_name + " does not exist and has to be generated once before using it. The command to generate it is: " + command)
        self.dset = f[self.dset_name]

    def __len__(self):
        return self.dset.shape[0]

    def __getitem__(self, item):
        out = self.dset[item]
        return torch.from_numpy(out).float()

    def __str__(self):
        return self.dset_name


def get_dataset_name(split, bars, stride, pianoroll):
    return ("pianoroll_" if pianoroll else "") + split + '_' + str(bars) + "bars_" + str(stride) + "stride_tempo_" + "computed" if calculate_correct_tempo else "120"

def create_final_dataset(split, bars=2, stride=1, pianoroll=False):

    dset_name = get_dataset_name(split, bars, stride, pianoroll)
    maestro = MaestroDataset(split, bars)

    with h5py.File(finalDataset_path, 'a') as f:
        if dset_name in f.keys():
            print("Final Dataset '" + dset_name + "' already exists")
            return

        c = 0
        for performance in maestro:

            c += 1
            print("\nperformance " + str(c) + " with " + str(performance.pianoroll.shape[0] // 16) + " bars")

            discarded = 0
            snippet_list = []

            # transpose performance
            for semitones in range(-6, 6):
                # print("transposing to " + str(semitones))
                transposed_performance = ppr.transpose(performance, semitones)
                if pianoroll:
                    transposed_performance = ppr.binarize(transposed_performance)
                    transposed_performance = transposed_performance.pianoroll
                    transposed_performance = transposed_performance[21:109]
                else:
                    transposed_performance = transposed_performance.pianoroll
                    transposed_performance = pianoroll_to_monophonic_repr(transposed_performance)

                for i in range(0, len(transposed_performance)-bars*resolution_per_beat*4, stride*resolution_per_beat*4):
                    i_end = i+bars*resolution_per_beat*4
                    snippet = transposed_performance[i:i_end]

                    # check for consecutive rests in performance (if performance is in monophonic representation -> also discards perfomances which have one note which is held very long
                    max_rests = 0
                    rests = 0
                    for slice in snippet:
                        if pianoroll and not np.any(slice):
                            rests += 1
                        elif not pianoroll and (slice[88] == 1 or slice[89] == 1):
                            rests += 1
                        else:
                            if rests > max_rests:
                                max_rests = rests
                            rests = 0
                    if rests > max_rests:
                        max_rests = rests

                    if max_rests > resolution_per_beat * 4:     # discard snippet because it has too many consecutive rests (more than 1 bar)
                        discarded += 1
                        continue

                    snippet_list.append(snippet)

            if len(snippet_list) > 120000:
                del snippet_list[120000:]
                print("throwing away some snippets for memory reasons, had " + len(snippet_list) + " snippets")
            if not snippet_list:
                print("\tdiscarded all " + str(discarded) + "snippets")
                continue
            performance_snippets = np.stack(snippet_list, axis=0)

            if dset_name not in f.keys():
                f.create_dataset(dset_name, data=performance_snippets, maxshape=(None, performance_snippets.shape[1], performance_snippets.shape[2]), compression='gzip')
            else:
                dset = f[dset_name]
                dset.resize(dset.shape[0] + performance_snippets.shape[0], axis=0)
                dset[-performance_snippets.shape[0]:] = performance_snippets

            print("\tDiscarded " + str(discarded) + " snippets and kept " + str(performance_snippets.shape[0]))
            del performance_snippets

        if dset_name not in f.keys():
            print("Dataset would be empty and is not generated. Maybe there are too many long pauses in the data which cause snippets to get discarded.")


def pianoroll_to_monophonic_repr(pianoroll):

    # also converts from dimension 128 to 90
    # using 88 for note-off and 89 for rest

    monophonic_repr = []                        # using a list for faster append() operation
    previous_slice = np.zeros(128)
    previous_no_note = True

    for slice in pianoroll:

        monophonic_slice = np.zeros(90, dtype=bool)
        no_note = True

        for k in range(108, 20, -1):            # looking only at pitches that exist on a piano keyboard ( [21, 108] )
            if slice[k] > 0:                    # note detected
                no_note = False
                if previous_slice[k] == 0:      # new note detected
                    monophonic_slice[k-21] = 1
                else:
                    monophonic_slice[89] = 1   # same note -> rest signal
                break                           # convert to monophonic by keeping only highest note

        if no_note:
            if previous_no_note:
                monophonic_slice[89] = 1       # ongoing rest -> rest signal
            else:
                monophonic_slice[88] = 1       # new rest -> note-off signal

        monophonic_repr.append(monophonic_slice)
        previous_slice = slice
        previous_no_note = no_note

    monophonic_repr = np.stack(monophonic_repr, axis=0)   # converting list to ndarray
    return monophonic_repr


def monophonic_repr_to_pianoroll(monophonic_repr):

    # also converts from dimension 90 to 128
    # using 88 for note-off and 89 for rest

    pianoroll = []  # using a list for faster append() operation
    previous_note = 89
    for slice in monophonic_repr:
        pianoroll_slice = np.zeros(128)
        for k in range(len(slice)-1, -1, -1):      # iterating in reverse order because rests (89) will occur relatively often
            if slice[k] > 0:  # note detected
                if k == 89:
                    if previous_note != 89:        # if last note is rest keep pianoroll at zero
                        pianoroll_slice[previous_note+21] = 1
                    continue
                if k == 88:
                    previous_note = 89             # interpreting note-off as rest
                    continue
                previous_note = k
                pianoroll_slice[k+21] = 1

        pianoroll.append(pianoroll_slice)

    pianoroll = np.stack(pianoroll, axis=0)  # converting list to ndarray
    return pianoroll



# old test code for conversion from/to monophonic representation

# performance = dataset.get_full_performance(0)
# performance = performance[0:500]
# performance[0, 0] = 1
# performance[1, 0] = 1
# performance[2, 0] = 1
#
# np.savetxt(fname="Pianoroll.txt", X=performance)
# performance2 = pianoroll_to_monophonic_repr(performance)
# np.savetxt(fname="monophonic.txt", X=performance2)
# performance3 = monophonic_repr_to_pianoroll(performance2)
# np.savetxt(fname="Pianoroll_2.txt", X=performance3)
# performance4 = pianoroll_to_monophonic_repr(performance3)

# ds = FinalDataset(split='test')
# item = ds.__getitem__(0)
# print(item)
