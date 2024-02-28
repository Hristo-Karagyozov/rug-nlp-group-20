import os
import shutil


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
    print(f"Dummy plotting")