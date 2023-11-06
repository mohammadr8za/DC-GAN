from os.path import join
import matplotlib.pyplot as plt
import os

import numpy as np
import torch
import pandas as pd


def checkpoint(loss_generator, loss_discriminator, root, Exp_ID, epoch,
               state_dict=None, plot_metrics=True, save_state=False, config=None):
    # Plot Metrics and Save Model Weights

    save_root = join(root, "Experiments", "train", Exp_ID)

    if plot_metrics:
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].plot(loss_generator)
        ax[0].set_title("generator loss")

        ax[1].plot(loss_discriminator)
        ax[1].set_title("discriminator loss")

        fig.suptitle("LOSS VISUALIZATION")
        fig.tight_layout()

        os.makedirs(join(save_root, "metrics"), exist_ok=True)
        plt.savefig(join(save_root, "metrics", "metrics_plot.png"))

        plt.close()

    if save_state:
        if epoch % 9 == 0:
            os.makedirs(join(save_root, "states"), exist_ok=True)
            torch.save(obj=state_dict, f=join(save_root, "states", "epoch_" + f"{epoch}" + ".pt"))

    if epoch == 0:
        config_df = pd.DataFrame(config, index=[0])
        config_df.to_csv(join(save_root, "config.csv"), index=False)

    print("Checkpoint!")

