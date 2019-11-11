import matplotlib.pyplot as plt
import pypianoroll as ppr
import numpy as np
import musicVAE
import math


# tum colors:
grey = (153/255, 153/255, 153/255)
green = (162/255, 173/255, 0.)
blue = (0/255, 101/255, 189/255)    # lighter blue: (100/255, 160/255, 200/255)
orange = (227/255, 114/255, 34/255)
black = (0., 0., 0.)
white = (1., 1., 1.)


# survey results
survey_complete = np.array([0, 0, 0, 0, 0, 3, 4, 6, 7, 6, 4, 4, 1, 1, 0, 0])
survey_hobby_musicians = np.array([0, 0, 0, 0, 0, 1, 3, 6, 7, 1, 3, 4, 0, 0, 0, 0])
survey_non_musicians = np.array([0, 0, 0, 0, 0 , 1, 1, 0, 0, 5, 1, 0, 0, 1, 0, 0])
survey_prof_musicians = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
survey_by_question = np.array([15, 25, 15, 20, 18, 24, 17, 25, 14, 14, 26, 28, 19, 19, 21])
x = np.array([i for i in range(16)])


def print_survey_info():
    texts = ["all participants", "professional musicians", "hobby musicians", "non-musicians"]
    surveys = [survey_complete, survey_prof_musicians, survey_hobby_musicians, survey_non_musicians]

    for t, s in zip(texts, surveys):
        answers = []
        for i in range(len(s)):
            n = s[i]
            for j in range(n):
                answers.append(i)

        participants = len(answers)
        mean = np.mean(answers)

        std = 0
        for n in answers:
            std += (n - mean) ** 2
        std /= participants     # computes the uncorrected std
        std = math.sqrt(std)
        print("Results for " + t + ":")
        print("\tParticipants: \t\t\t" + str(participants))
        print("\tMean: \t\t\t\t\t" + str(mean))
        print("\tStandard deviation:\t\t" + str(std) + "\n")


def plot_survey():
    #x = range(16)
    plt.bar(x, survey_prof_musicians, label="professional musicians", color=green)
    plt.bar(x, survey_hobby_musicians, bottom=survey_prof_musicians, label="hobby musicians", color=blue)
    plt.bar(x, survey_non_musicians, bottom=survey_prof_musicians+survey_hobby_musicians, label="non-musicians", color=orange)

    plt.ylabel("Number of Participants")
    plt.xlabel("Correct Answers")
    plt.ylim((0, 8))
    plt.xlim((0, 15))
    plt.xticks(x)
    plt.legend()
    plt.grid(True, axis="y")
    plt.savefig("survey.png")

def plot_survey_by_question():
    x_1 = x[1:]
    y = (survey_by_question / 36) * 100 # convert to percent
    plt.bar(x_1, y, color=blue)
    plt.ylabel("Percentage of correct answers")
    plt.xlabel("Question number")
    plt.ylim((0, 100))
    plt.xlim((0, 15))
    plt.xticks(x)

    plt.grid(True, axis="y")
    plt.savefig("survey_by_question.png")




def plot_train_and_eval_loss(checkpoint_name, location="/home/micaltu/tss19-VAE-music-generation/Models/Checkpoints"):
    checkpoint = musicVAE.get_checkpoint(checkpoint_name, location)
    loss = checkpoint['last_loss_list']
    train_loss = [99] + [l[0] for l in loss]
    eval_loss = [99] + [l[1] for l in loss]
    loss = None

    x = range(1, len(train_loss)+1)


    plt.plot(x, train_loss, '--', color=blue, label="training loss")
    plt.plot(x, eval_loss, color=blue, label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.ylim((0, None))
    plt.savefig(checkpoint_name + "_loss.png")


def plot_all_2bar_losses(location="/home/micaltu/tss19-VAE-music-generation/Models/Checkpoints"):
    names = ["2bars_3stride_after_epoch_8", "pianoroll_train_2bars_3stride_tempo_computed_after_epoch_6", "train_2bars_1stride_tempo_computed_transposed_after_epoch_35", "pianoroll_train_2bars_1stride_tempo_computed_transposed_after_epoch_28"]
    legend_texts = ["MIDI-like and all keys", "piano roll and all keys", "MIDI-like and one key", "piano roll and one key"]
    colors = [blue, orange, green, grey]#['r', 'g', 'tab:purple', 'tab:cyan']   #
    losses = []
    for n in names:
        checkpoint = musicVAE.get_checkpoint(n, location)
        loss = checkpoint['last_loss_list']
        eval_loss = [99.] + [l[1] for l in loss]



        if n == "2bars_3stride_after_epoch_8": # normalize first loss
            factor = 24.090449431366114/eval_loss[-1]

            old_loss = eval_loss
            eval_loss = [l * factor for l in old_loss]
            eval_loss[0] = 99.

        loss = None

        gradient_updates = [x for x in range(0, 27055, 27055//len(eval_loss))]
        gradient_updates = gradient_updates[0:len(eval_loss)]

        losses += [(gradient_updates, eval_loss)]

    for i in range(4):
        plt.plot(losses[i][0], losses[i][1], color=colors[i], label=legend_texts[i])

    # should be exactly the same, but strange things happen:
    #
    # for (x, loss), c, l in zip(losses, colors, legend_texts):
    #     plt.plot(x, loss, c, label=l)

    plt.legend()

    # for showing a custom legend with text in the line colors:
    #
    # x_pos = x[len(x)//2]
    # plt.text(x_pos, 90, "MIDI-like and all keys", color=colors[0])
    # plt.text(x_pos, 85, "pianoroll and all keys", color=colors[1])
    # plt.text(x_pos, 80, "MIDI-like and one key", color=colors[2])
    # plt.text(x_pos, 75, "pianoroll and one key", color=colors[3])

    plt.xlabel("gradient updates")
    plt.ylabel("evaluation loss")
    plt.savefig("2bar_losses.png")


def plot_loss_all_lengths(location="/home/micaltu/tss19-VAE-music-generation/Models/Checkpoints"):
    fig, axs = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True, constrained_layout=True)
    names = ["pianoroll_train_2bars_1stride_tempo_computed_transposed_after_epoch_94", "pianoroll_train_4bars_1stride_tempo_computed_transposed_after_e_21", "pianoroll_train_8bars_1stride_tempo_computed_transposed_after_e_11", "pianoroll_train_16bars_1stride_tempo_computed_transposed_after_epoch_9"]

    bars = 2
    for ax, checkpoint_name in zip(axs.flat, names):
        checkpoint = musicVAE.get_checkpoint(checkpoint_name, location)
        loss = checkpoint['last_loss_list']
        train_loss = [99] + [l[0] for l in loss]
        eval_loss = [99] + [l[1] for l in loss]
        loss = None

        x = range(1, len(train_loss) + 1)
        if bars == 16:
            ax.plot(x, train_loss, '--', color=orange, label="training loss")
            ax.plot(x, eval_loss, color=blue, label="validation loss")
            ax.legend(loc="lower left")
        else:
            ax.plot(x, train_loss, '--', color=orange)
            ax.plot(x, eval_loss, color=blue)
        ax.set_xlabel("epochs", fontsize=12)
        ax.set_ylim((0, None))
        if bars == 2:
            ax.set_ylabel("loss", fontsize=12)
        ax.set_title(str(bars) + " bars", fontsize=20)

        bars *= 2
    fig.savefig("all_losses.png") # dpi=100


def plot_pianoroll(path="Sampled/4_bar_samples/sample_4_0.midi"):
    midi = ppr.parse(filepath=path, beat_resolution=4)  # get Multitrack object
    midi = midi.tracks[0]  # get first/only track
    pianoroll = midi.pianoroll
    print(pianoroll.shape)

    ppr.plot(midi, filename="testppr.png", beat_resolution=4)

#the melodies are not very recognizable in the plot
def plot_interpolation_sotw_to_lick():
    interpolations = []
    for i in range(0, 10, 3):
        path = "Sampled/interpolation_SotW_to_lick/2/interpolate_" + str(i) + ".midi"
        #plot_pianoroll(path)
        #return

        midi = ppr.parse(filepath=path, beat_resolution=4)  # get Multitrack object
        midi = midi.tracks[0]  # get first/only track
        midi.name = ""
        if i == 0:
            midi.name = "start sequence"
        if i == 10:
            midi.name = "end sequence"

        interpolations.append(midi)

    mt = ppr.Multitrack(tracks=interpolations, beat_resolution=4)
    p, _ = ppr.plot(mt, yticklabel="off", xtick='beat', xticklabel=True, grid="both")
    #p.set_size_inches((8, 8), forward=True)

    filename = "interpolation_SotW_to_lick.png"
    p.savefig(filename)


def plot_interpolation_pianorolls(bars=16):
    interpolations = []
    for i in range(0, 6, 2):
        path = "Sampled/" + str(bars) + "bar_interpolation/interpolate_" + str(i) + ".midi"

        midi = ppr.parse(filepath=path, beat_resolution=4)  # get Multitrack object
        midi = midi.tracks[0]  # get first/only track
        midi.name = ""
        if i == 0:
            midi.name = "start sequence"
        if i == 4:
            midi.name = "end sequence"
        pr = midi.pianoroll

        # padding to full length in case MIDI file ends earlier
        if pr.shape[0] != bars * 16:
            padding = np.zeros((bars * 16 - pr.shape[0], pr.shape[1]))
            pr = np.concatenate((pr, padding), axis=0)
            midi.pianoroll = pr
        interpolations.append(midi)

    mt = ppr.Multitrack(tracks=interpolations, beat_resolution=4)

    if bars == 16:
        p, _ = ppr.plot(mt, yticklabel="number", xtick='beat', xticklabel=False, grid="off")
        # there seems to be a bug in ppr, despite xticklabel=False, the plot still has the labels for each x-axis value

    else:
        p, _ = ppr.plot(mt,yticklabel="number", xtick='beat', xticklabel=True, grid="both")
    p.set_size_inches((8, 8), forward=True)

    filename = str(bars) + "bar_interpolation.png"
    p.savefig(filename)




# uncomment what you want to plot:

# plot_all_2bar_losses()
# plot_train_and_eval_loss("pianoroll_train_8bars_1stride_tempo_computed_transposed_after_e_11")
# plot_loss_all_lengths()
# plot_interpolation_pianorolls(4)
plot_survey()
# plot_survey_by_question()
# print_survey_info()

