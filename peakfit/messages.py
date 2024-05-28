"""Contain IO messages."""

from __future__ import annotations

from pathlib import Path

from lmfit.minimizer import MinimizerResult
from lmfit.printfuncs import fit_report
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from peakfit import __version__

console = Console(record=True)


LOGO = r"""
   ___           _      ___ _ _
  / _ \___  __ _| | __ / __(_) |_
 / /_)/ _ \/ _` | |/ // _\ | | __|
/ ___/  __/ (_| |   </ /   | | |_
\/    \___|\__,_|_|\_\/    |_|\__|

"""


def print_logo() -> None:
    """Display the logo in the terminal."""
    logo = Text(LOGO, style="blue")
    description = "Perform peak integration in  \npseudo-3D spectra\n\n"
    version = "Version: "
    version_number = Text(f"{__version__}", style="red")
    all_text = Text.assemble(logo, description, version, version_number)
    panel = Panel.fit(all_text)
    console.print(panel)


def print_fitting() -> None:
    """Print the fitting message."""
    message = "\n — Fitting peaks..."
    console.print(message, style="bold yellow")


def print_peaks(peaks) -> None:
    """Print the peak names that are being fitted."""
    peak_list = ", ".join(f"{name:s}" for name in peaks["name"])
    message = f"Peak(s): {peak_list}"
    panel = Panel.fit(message, style="green")
    console.print(panel)


def print_segmenting() -> None:
    """Print the segmenting message."""
    message = "\n — Segmenting the spectra and clustering the peaks..."
    console.print(message, style="bold yellow")


def print_fit_report(minimizer_result: MinimizerResult) -> None:
    """Print the fitting report."""
    console.print("\n", Text(fit_report(minimizer_result, min_correl=0.5)), "\n")


def export_html(filehtml: Path) -> None:
    filehtml.write_text(console.export_html())


def print_reading_files() -> None:
    """Print the message for reading files."""
    message = "\n — Reading files..."
    console.print(message, style="bold yellow")


def print_plotting() -> None:
    """Print the message for plotting."""
    filename = "[bold green]profiles.pdf[/]"
    message = f"\n[bold yellow] — Plotting to[/] {filename}[bold yellow]...[/]"
    console.print(Text.from_markup(message))


def print_filename(filename: Path) -> None:
    """Print the filename."""
    message = f"    ‣ [green]{filename}[/]"
    console.print(Text.from_markup(message))


def print_estimated_noise(noise: float) -> None:
    """Print the estimated noise."""
    message = f"\n [bold yellow]— Estimated noise:[/] [bold green]{noise:.2f}[/]"
    console.print(Text.from_markup(message))
