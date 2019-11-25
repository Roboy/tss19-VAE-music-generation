# Generating Musical Note Sequences with Variational Autoencoders

A bachelor's thesis in informatics at TUM (Technical University of Munich)

**Abstract:**

>In recent years, many new generative machine learning approaches for symbolic music (musical notes, as opposed to sounds) have been published. They use a wide variety of network architectures to generate musical notes and different ways to represent these notes. However, there is not yet any consensus or guideline on how symbolic music is best represented and whether transposing all training sequences to the same key or to all possible keys leads to better results. One contributing factor is that most authors choose one option and train their models with it, not addressing the question whether a different choice could improve the quality of the generated music or speed up the training process. To provide some insight into this question, we reimplement a recent architecture, the MusicVAE, and train it on multiple datasets of the same data, but treated and represented differently in each one. We then show that the two modifications we made compared to the original authors do indeed improve both the training time and the resulting performance of our implementation. We reach a similar performance to the original implementation on 2-bar long sequences and conduct a blind listening test, which confirms that the generated music is indistinguishable from human-made music. Not only does this show which options work best for this specific architecture, it also demonstrates that these choices can have significant impact on the performance of generative neural networks, and that there is a need for further comparison, also including other architectures than the Variational Autoencoder we used.

The blind listening test was done online and is still available [here](https://forms.gle/Wd5hN5wQqooL6cvY6).

## Description

This repository contains the code written for the thesis, a reimplementation of the [MusicVAE](https://magenta.tensorflow.org/music-vae) from the Google Magenta team. It can be used to generate new music and to interpolate between existing or randomly generated music sequences with a command line interface (CLI). Furthermore, all the code to generate training datasets in various configurations and to train the model is also accessible from the CLI.

This version supports only monophonic melodies. It was trained on music sequences with a length of 2, 4, 8, and 16 bars. For the 2-bar sequences, all possible combinations of the following data representations/treatments were used:
- Transposing all training sequences to C major/a minor or to all possible keys
- Using a standard pianoroll representation with an additional rest token or using the same, more MIDI-like data representation from the magenta team (described [here](https://arxiv.org/abs/1803.05428) or in my thesis).

All training sequences were generated from the [MAESTRO dataset](https://magenta.tensorflow.org/datasets/maestro).

## Listening to some generated samples
If you only want to listen to some samples generated with this code, have a look at the Sampled folder. The examples included in the thesis are always the first ones, e.g., `Sampled/16bar_samples/sample_16_0.midi`.
A particularly good example of what this kind of model can be used for is the interpolation between the riff from Smoke on the Water by Deep Purple and a commonly used jazz lick. It can be found in `Sampled/interpolation_examples/SotW_to_lick`.

If you want to create your own music, just keep on reading.

## Installation

1. All the following paths will be relative to the root folder of this project (tss19-VAE-music-generation), so after cloning the repository,  navigate to the root directory (replace '/path/to' with the correct path on your machine).
```
cd /path/to/tss19-VAE-music-generation
```

2. Create a new virtual environment above the root folder of the project. You can replace venv with any other name.
```bash
virtualenv -p python3 ../venv
```
3. Activate the virtual environment you have just created (at the end, you can deactivate it again with `deactivate`).
```bash  
source ../venv/bin/activate  
```
4. Install all the necessary dependencies.
```bash  
pip install -r requirements.txt
```

(Optional) If you want to run the model on your GPU, you will need to [install CUDA](https://pytorch.org/get-started/locally/) on your system. If CUDA is installed and your hardware is supported, everything will be run on the GPU automatically.

## Downloading the training data
This is only necessary if you want to train the model yourself.
Download the HDF file containing all the necessary datasets from [here](https://drive.google.com/open?id=1z1AzPuEL8I4SSJCzG2RnO3Xp5ORTBWGo) and save it under `Data/FinalDataset/final_dataset.h5`.
## Downloading the MAESTRO Dataset
In case you want to generate new data sets, you will need to download the MAESTRO dataset. Otherwise, you can skip this step.

1. Download the MAESTRO Dataset.
```bash
wget https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip
```

2. Unpack it into the correct directory.
```bash
unzip maestro-v2.0.0-midi.zip -d Data/
```


## Downloading Pretrained Models
Pretrained models can be downloaded from [here](https://drive.google.com/open?id=18NdFzZqUMGpcQ3-6WFYjEHmcVvPYvdIC). They are named the same as the dataset they were trained with. After downloading one or more of them, save them to the folder `Models/Final`.
Training checkpoints are saved in `Models/Checkpoints`, so if you want to resume the training with any of the checkpoints you will have to move or copy them there.
## Using the Command Line Interface
To use the CLI, the virtual environment you created during the installation needs to be activated. If it is not anymore, repeat step 3 of the installation.

The CLI supports a wide range of operations. To see all the possible commands, have a look at the help.
```bash
python musicVAE.py -h
```
Each one of the commands has its own help that explains its parameters (replace \<command> with the command you want to know more about).
```bash
python musicVAE.py <command> -h
```
In general, you have to choose which of the models to use. How many bars, piano roll or MIDI-like representation, transposed to C major/a minor or to all keys etc. The models trained with sequences in piano roll representation and transposed to C major/a minor have the best performance. Other options are only available for 2-bar sequences. Therefore, for best results and for longer sequences, you have to use the -p and -t flags.

#### Example use cases:
- Generating a 4-bar long music sequence. The result will be saved as `Sampled/sample.midi`. The flags -p and -t can also be shortened to -pt.
```bash  
python musicVAE.py sample 4 -p -t  
```
- Interpolating between a randomly generated 2-bar sequence and the 2-bar long sequence `Sampled/manually_created_sequences/smoke.midi` (the first few notes of the riff from Smoke on the Water by Deep Purple) and generating 3 in-between sequences.
```bash  
python musicVAE.py interpolate 2 3 -ptv -e /home/micaltu/tss19-VAE-music-generation/Sampled/manually_created_sequences/smoke.midi 
```
- Reconstructing the Smoke on the Water riff 5 times to get an imperfect reconstruction. This can be done to get a slightly different version of a music sequence.
```bash  
python musicVAE.py reconstruct 2 -ptv -n 5 --start_sequence=Sampled/manually_created_sequences/smoke.midi
```
