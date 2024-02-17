import os
import time
from glob import glob


timestr = lambda: time.strftime("%Y%m%d-%H%M%S")
DATA_PATH = "data/user_doc"


def get_file_path(file_name):
    file_path = os.path.abspath(file_name)  # Join the directory and file name
    return file_path


def get_flie_path_from_name(file_name):
    files = glob(os.path.join(DATA_PATH, "**", file_name), recursive=True)
    return get_file_path(files[0])
