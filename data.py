import numpy as np
import torch
import pypianoroll as ppr
import pandas
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import pretty_midi
import music21
import os
from inspect import getsourcefile
from os.path import abspath
import random
import sys
import h5py

# get current path
path_to_root = abspath(getsourcefile(lambda:0))

# remove filename from path
name = os.path.basename(path_to_root)
end_index = len(path_to_root) - len(name)
path_to_root = path_to_root[0:end_index]

finalDataset_path = path_to_root + "Data/FinalDataset/final_dataset.h5"

maestro_root_dir = path_to_root + "Data/maestro-v2.0.0"
maestro_csv_path = maestro_root_dir + "/maestro-v2.0.0.csv"

# Parameters for generating data from performances
calculate_correct_tempo = True
false_tempo = 120           # the tempo assigned to every one of the midi files
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


def delete_dataset(split, bars, stride, pianoroll, transposed):
    dset_name = get_dataset_name(split, bars, stride, pianoroll, transposed)
    with h5py.File(finalDataset_path, 'a') as f:
        if dset_name in f.keys():
            del f[dset_name]
            print("deleted dataset " + dset_name)
        else:
            print("dataset " + dset_name + " does not exist")

'''
about Maestro Dataset:

Tempo of the MIDI files is always 120 and downbeat in ppr Multitrack object is only True at index 0, rest is false

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
            # would also be possible to drop more information like title, but might be handy for debugging

            self.csv_data_frame = self.csv_data_frame[self.csv_data_frame['split'] == split]
            self.csv_data_frame.set_axis(range(len(self.csv_data_frame)), axis='index', inplace=True)

            self.compute_tempo()
            self.csv_data_frame.to_csv(path_or_buf=edited_csv_path, index=False, na_rep="NaN")

            self.csv_data_frame = pandas.read_csv(edited_csv_path)

        if 'key' not in self.csv_data_frame.columns:
            self.compute_key()
            self.csv_data_frame.to_csv(path_or_buf=edited_csv_path, index=False)

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

    def get_key(self, index):
        if 'key' not in self.csv_data_frame.columns:
            raise ValueError("key information seems to not have been generated yet. Call MaestroDataset.compute_key() to compute it and save it in the csv file.")

        row = self.csv_data_frame.iloc[[index]]
        key = row['key']
        key = key.item()        # TODO change to not deprecated function when the docs specify alternative
        key = key[0:2]
        return key

    def get_path(self, index):
        row = self.csv_data_frame.iloc[[index]]
        out = row['midi_filename']
        out = out[index]
        return out

    # returns a ppr.Track object (because then ppr.transpose() can only be called with tracks, not with pianorolls)
    def get_full_performance(self, index, beat_resolution=resolution_per_beat):
        infos = self.csv_data_frame.iloc[index]
        midi_filename = infos['midi_filename']
        path = "" + self.root_dir + "/" + midi_filename

        midi = ppr.parse(filepath=path, beat_resolution=beat_resolution)  # get Multitrack object
        midi = midi.tracks[0]                                # get first/only track
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

    def compute_key(self):
        print("computing the keys of the performances...")
        key_list = []
        for _, row in self.csv_data_frame.iterrows():       #iterrows returns index and row as pandas Series object
            path = "" + self.root_dir + "/" + row['midi_filename']
            midi_data = music21.converter.parse(path)
            key = midi_data.analyze('key')      # checked first 5 performances, and key was always right, even with the first performance in dataset which uses a lot of whole tone scales and chromaticism (meaning guessing its correct key should be quite hard)
            key_list.append(key)
            print(key, '\t\t', path)

        self.csv_data_frame = self.csv_data_frame.assign(key=pandas.Series(key_list))


class FinalDataset(Dataset):
    def __init__(self, split='train', bars=2, stride=1, pianoroll=True, transpose=True):
        if (split != 'train' and split != 'test' and split != 'validation'):
            raise ValueError("class FinalDataset was initialized with invalid aruments, split has to be 'train', 'test' or 'validation'!")
        self.split = split
        self.bars = bars
        self.stride = stride
        self.dset = None
        self.dset_name = get_dataset_name(split, bars, stride, pianoroll, transpose)

        f = h5py.File(finalDataset_path, 'r')        # f stays opened, a __del__ method would be an option but seems to be unreliable/ontroversial. A with-statement doesnt work because f needs to stay open as long as dset is used
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


def get_dataset_name(split, bars, stride, pianoroll, transpose):
    return ("pianoroll_" if pianoroll else "") + split + '_' + str(bars) + "bars_" + str(stride) + "stride_tempo_" + ("computed" if calculate_correct_tempo else "120") + ("_transposed" if transpose else "")


def create_final_dataset(split, bars=2, stride=1, pianoroll=False, transpose=False):

    dset_name = get_dataset_name(split, bars, stride, pianoroll, transpose)
    maestro = MaestroDataset(split, bars)

    transpose_dict = {'C ': 0, 'C#': -1, 'D-': -1, 'D ': -2, 'D#': -3, 'E-': -3, 'E ': -4, 'F-': -4, 'E#': -5, 'F ': -5, 'F#': -6, 'G-': -6, 'G ': -7, 'G#': -8, 'A-': -8, 'A ': -9, 'A#': -10, 'B-': -10, 'B ': -11, 'C-': -11, 'B#': -11,       # major keys (capital letters)
                      'a ': 0, 'a#': -1, 'b-': -1, 'b ': -2, 'c-': -2, 'b#': -3, 'c ': -3, 'c#': -4, 'd-': -4, 'd ': -5, 'd#': -6, 'e-': -6, 'e ': -7, 'f-': -7, 'e#': -8, 'f ': -8, 'f#': -9, 'g-': -9, 'g ': -10, 'g#': -11, 'a-': -11}      # minor keys

    with h5py.File(finalDataset_path, 'a') as f:
        if dset_name in f.keys():
            print("Final Dataset '" + dset_name + "' already exists")
            return

        c = 0
        for index in range(len(maestro)):
            performance = maestro.__getitem__(index)

            c += 1
            print("\nperformance " + str(c) + " with " + str(performance.pianoroll.shape[0] // 16) + " bars")

            discarded = 0
            snippet_list = []

            # transpose performance
            t = None
            if not transpose:
                t = range(-6, 6)
            else:
                key = maestro.get_key(index)
                distance = transpose_dict[key]
                if distance < -6:
                    distance += 12
                t = [distance]

            for semitones in t:
                # print("key is {}, transposing by {} semitones".format((key if transpose else "/"), semitones))
                transposed_performance = ppr.transpose(performance, semitones)
                if pianoroll:
                    transposed_performance = ppr.binarize(transposed_performance)
                    transposed_performance = transposed_performance.pianoroll
                    transposed_performance = transposed_performance[:, 21:109]      # using only pitches that exist on a piano keyboard ( [21, 108] )
                    transposed_performance = pianoroll_to_mono_pianoroll(transposed_performance)
                else:
                    transposed_performance = transposed_performance.pianoroll
                    transposed_performance = pianoroll_to_monophonic_repr(transposed_performance)

                for i in range(0, len(transposed_performance)-bars*resolution_per_beat*4, stride*resolution_per_beat*4):
                    i_end = i + bars * resolution_per_beat * 4
                    snippet = transposed_performance[i:i_end]

                    if pianoroll:
                        pauses = np.zeros((snippet.shape[0], 1), dtype=bool)

                    # check for consecutive rests in performance (if performance is in monophonic representation -> also discards perfomances which have one note which is held very long
                    max_rests = 0
                    rests = 0
                    position = 0
                    for slice in snippet:
                        if pianoroll and not np.any(slice):
                            rests += 1
                            pauses[position, 0] = 1
                        elif not pianoroll and (slice[88] == 1 or slice[89] == 1):
                            rests += 1
                        else:
                            if rests > max_rests:
                                max_rests = rests
                            rests = 0

                        position += 1

                    if rests > max_rests:
                        max_rests = rests

                    if max_rests > resolution_per_beat * 4:     # discard snippet because it has too many consecutive rests (more than 1 bar)
                        discarded += 1
                        continue

                    if pianoroll:
                        snippet = np.concatenate((snippet, pauses), axis=1)

                    snippet_list.append(snippet)

            if len(snippet_list) > 120000:      # longer lists lead to memory error during the stack operation a few lines below; value found by experimenting, probably depends on machine
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


def get_random_training_data(amount=15, split="train", bars=2, stride=1, pianoroll=True, transpose=True):
    dset = FinalDataset(split, bars, stride, pianoroll, transpose)
    it = random.sample(range(0, len(dset)), amount)
    n = 0
    for i in it:
        snippet = dset.__getitem__(i)
        if not pianoroll:
            snippet = monophonic_repr_to_pianoroll(snippet)
        filename = "Sampled/training_sample_" + str(n) + ".midi"
        pianoroll_to_midi(snippet, filename)
        n += 1




# works with input of all sizes, but input has to be mono already!
def pianoroll_to_one_hot_pianoroll(pianoroll):
    pauses = np.zeros((pianoroll.shape[0], 1), dtype=bool)
    position = 0
    for slice in pianoroll:
        if not np.any(slice):
            pauses[position, 0] = 1
        position +=1

    pianoroll = np.concatenate((pianoroll, pauses), axis=1)
    return pianoroll


def one_hot_pianoroll_to_small_pianoroll(pianoroll):
    return pianoroll[..., 0:88]


#  works with input of all sizes, discards all but highest notes for every time step
def pianoroll_to_mono_pianoroll(pianoroll):
    monophonic = []  # using a list for faster append() operation

    for slice in pianoroll:

        monophonic_slice = np.zeros(len(slice), dtype=bool)

        if slice.any():
            for k in range(len(monophonic_slice)-1, -1, -1):
                if slice[k]:                    # note detected
                    monophonic_slice[k] = 1
                    break                       # convert to monophonic by keeping only highest note

        monophonic.append(monophonic_slice)

    monophonic = np.stack(monophonic, axis=0)  # converting list to ndarray
    return monophonic


# input of size (t, 128), output (t, 90)
def pianoroll_to_monophonic_repr(pianoroll):

    # also converts from dimension 128 to 90
    # using 88 for note-off and 89 for rest

    monophonic_repr = []                        # using a list for faster append() operation
    previous_slice = np.zeros(128)
    previous_no_note = True

    for slice in pianoroll:

        monophonic_slice = np.zeros(90, dtype=bool)
        no_note = True

        # possible improvement: use slice.any() before to check if there are any notes in the slice at all (see pianoroll_to_mono_pianoroll() above)
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


def model_output_to_pianoroll(sample, pianoroll=False):
    if sample.dim() == 3:
        if sample.shape[0] == 1:
            sample = sample.squeeze()
        else:
            raise ValueError("model output has a batch size bigger than one")
    sample = sample.argmax(dim=1)
    num_classes = 89 if pianoroll else 90
    sample = F.one_hot(sample, num_classes)

    if not pianoroll:
        sample = monophonic_repr_to_pianoroll(sample)
    else:
        sample = sample[:, 0:88]   #remove the pause tokens

    return sample


def small_to_full_pianoroll(snippet):
    if torch.is_tensor(snippet):
        lower_notes = torch.zeros((snippet.shape[0], 21), dtype=snippet.dtype)
        higher_notes = torch.zeros((snippet.shape[0], 19), dtype=snippet.dtype)
        snippet = torch.cat((lower_notes, snippet, higher_notes), axis=1)
    else:
        lower_notes = np.zeros((snippet.shape[0], 21), dtype=snippet.dtype)
        higher_notes = np.zeros((snippet.shape[0], 19), dtype=snippet.dtype)
        snippet = np.concatenate((lower_notes, snippet, higher_notes), axis=1)

    return snippet


def full_to_small_pianoroll(snippet):
    snippet = snippet[..., 21:109]
    return snippet


# works with all kinds of pianorolls
def pianoroll_to_midi(snippet, filename="Sampled/sample.midi"):
    snippet = np.asarray(snippet, dtype=np.uint8)
    snippet = snippet * 127  # sets velocity of notes from 1 to 127 (max MIDI velocity)

    if snippet.shape[1] == 89:
        snippet = one_hot_pianoroll_to_small_pianoroll(snippet)
        snippet = small_to_full_pianoroll(snippet)
    elif snippet.shape[1] == 88:
        snippet = small_to_full_pianoroll(snippet)
    else:
        if not snippet.shape[1] == 128:
            raise ValueError("input shape does not have 128 pitches (or 88, then it will be converted automatically) and cannot be converted to MIDI!")

    snippet = ppr.Track(pianoroll=snippet)
    snippet = ppr.Multitrack(tracks=[snippet], tempo=120, beat_resolution=4)
    ppr.write(snippet, path_to_root + filename)


def midi_to_small_one_hot_pianoroll(path, beat_resolution=4):
    midi = ppr.parse(filepath=path, beat_resolution=beat_resolution)  # get Multitrack object
    midi = midi.tracks[0]
    midi = ppr.binarize(midi)
    midi = midi.pianoroll
    midi = full_to_small_pianoroll(midi)
    midi = pianoroll_to_mono_pianoroll(midi)
    midi = pianoroll_to_one_hot_pianoroll(midi)
    midi = torch.from_numpy(midi).float()
    return midi
