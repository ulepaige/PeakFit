import errno
import os.path
import sys


def make_dirs(path=None):
    """Make the directory if needed"""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise


def print_peaks(peaks, files=None):
    if files is None:
        files = (sys.stdout,)

    message = "*  Peak(s): "
    message += ", ".join(f"{peak[0]:s}" for peak in peaks)
    message += "  *"
    stars = "*" * len(message)
    message = "\n".join([stars, message, stars])

    for file in files:
        print(message, end="\n\n", file=file)


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))
