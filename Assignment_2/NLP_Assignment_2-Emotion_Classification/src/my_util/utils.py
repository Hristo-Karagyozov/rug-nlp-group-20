import os
import shutil
from collections import namedtuple
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorboard import program


Bar = namedtuple("Bar", ("mean", "std", "tag"))
Curve = namedtuple("Curve", ("time", "mean", "std", "tag"))


class Metrics:
    def __init__(self, true_labels, predicted_labels):
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels

    def confusion_matrix(self):
        return confusion_matrix(self.true_labels, self.predicted_labels)

    def accuracy(self):
        return accuracy_score(self.true_labels, self.predicted_labels)

    def precision(self):
        return precision_score(self.true_labels, self.predicted_labels, average='weighted')

    def f1_score(self):
        return f1_score(self.true_labels, self.predicted_labels, average='weighted')


def LossCurve(mean=np.array([]), time=np.array([]), std=np.array([]), tag="Curve"):
    """
    Parameters
    ----------
    time : np.ndarray
    mean : np.ndarray
    std : np.ndarray
    tag : str
    Returns
    -------
    Curve
    """
    if len(time) == 0 and len(mean) != 0:
        time = np.array([x for x, _ in enumerate(mean)])
    if len(std) == 0 and len(mean) != 0:
        std = np.zeros(len(mean))
    return Curve(time, mean, std, tag)


def copy_file(source_file, destination_dir):
    """
    Copies a file from the current location to the desired destination. Creates the destination directory if it does
    not yet exist.

    Parameters
    ----------
    source_file : str
        Current location of the file.
    destination_dir : str
        Destination location of the file.
    """

    os.makedirs(destination_dir, exist_ok=True)
    destination_path = os.path.join(destination_dir, os.path.basename(source_file))
    shutil.copy(source_file, destination_path)


def plot_curves(curves, save_path=None, name="", save_data=False):
    """
    Plots a list of curves in one figure.

    Parameters
    ----------
    curves : list
    save_path : str
    name : str
    save_data : bool
    """
    colors = plt.cm.get_cmap('tab10')

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        if save_data:
            os.makedirs(os.path.join(save_path, "curve_data"), exist_ok=True)

    plt.rcParams['font.family'] = 'serif'
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--')

    for idx, curve in enumerate(curves):
        plt.plot(curve.time, curve.mean, label=curve.tag, color=colors(idx))
        if save_path is not None and save_data:
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_time"), curve.time)
            np.save(os.path.join(save_path, "curve_data", f"{name}_{curve.tag}_mean"), curve.mean)

    plt.subplots_adjust(left=0.15, right=0.9, bottom=0.15, top=0.9)
    plt.tight_layout()
    plt.legend()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"{name}"))
    else:
        plt.show()
    plt.close()


def open_tensorboard(log_dir):
    """
    Parameters
    ----------
    log_dir : str
    """
    tb = program.TensorBoard()
    os.makedirs(log_dir, exist_ok=True)
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")