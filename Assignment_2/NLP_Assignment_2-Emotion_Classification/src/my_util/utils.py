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


def explore_data(data, save_path=None):
    """
    Generates some plots to visualize the data.

    Parameters
    ----------
    data : pandas.DataFrame
    save_path : str
        Directory where the plots are saved (None means no saving)
    """
    pass