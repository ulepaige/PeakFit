import sys


def print_peaks(peaks, files=None):
    if files is None:
        files = (sys.stdout,)

    peak_list = ", ".join(f"{peak[0]:s}" for peak in peaks)
    message = f"*  Peak(s): {peak_list}  *"
    stars = "*" * len(message)
    message = "\n".join([stars, message, stars])

    for file in files:
        print(message, end="\n\n", file=file)


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))
