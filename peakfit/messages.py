"""Contain IO messages"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from peakfit import __version__

console = Console()


LOGO = "\n".join(
    [
        r"   ___           _      ___ _ _",
        r"  / _ \___  __ _| | __ / __(_) |_",
        r" / /_)/ _ \/ _` | |/ // _\ | | __|",
        r"/ ___/  __/ (_| |   </ /   | | |_",
        r"\/    \___|\__,_|_|\_\/    |_|\__|",
        "",
        "",
    ]
)


def print_logo() -> None:
    """Display the logo in the terminal"""
    logo = Text(LOGO, style="blue")
    description = "Perform peak integration in  \npseudo-3D spectra\n\n"
    version = "Version: "
    version_number = Text(f"{__version__}", style="red")
    all_text = Text.assemble(logo, description, version, version_number)
    panel = Panel.fit(all_text)
    console.print(panel)


def print_peaks(peaks, files=None):
    """Print the peak names that are being fitted"""

    if files is None:
        files = (sys.stdout,)

    peak_list = ", ".join(f"{peak[0]:s}" for peak in peaks)
    message = f"*  Peak(s): {peak_list}  *"
    stars = "*" * len(message)
    message = "\n".join([stars, message, stars])

    for file in files:
        print(message, end="\n\n", file=file)
