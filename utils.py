import matplotlib.pyplot as plt
import numpy as np
import musicVAE

#tum colors:
grey = (153/255, 153/255, 153/255)
green = (162/255, 173/255, 0.)
blue = (0/255, 101/255, 189/255)    # lighter blue: (100/255, 160/255, 200/255)
orange = (227/255, 114/255, 34/255)
black = (0., 0., 0.)
white = (1., 1., 1.)


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
    fig.savefig("all_losses.png")


# uncomment what you want to plot

#plot_all_2bar_losses()
#plot_train_and_eval_loss("pianoroll_train_8bars_1stride_tempo_computed_transposed_after_e_11")
#plot_loss_all_lengths()