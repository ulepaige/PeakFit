"""Main module."""

from collections.abc import Sequence
from pathlib import Path

import lmfit as lf
import nmrglue as ng
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from peakfit.cli import build_parser
from peakfit.clustering import Cluster, Spectra, cluster_peaks
from peakfit.computing import calculate_shape_heights, residuals, simulate_data
from peakfit.messages import (
    export_html,
    print_estimated_noise,
    print_fit_report,
    print_fitting,
    print_logo,
    print_peaks,
)
from peakfit.noise import estimate_noise
from peakfit.peaklist import read_list
from peakfit.plots import plot_pdf
from peakfit.shapes import create_params


def read_spectra(
    paths_spectra: Sequence[Path],
    paths_z_values: Sequence[Path],
    exclude_list: Sequence[int],
) -> Spectra:
    """Read NMRPipe spectra and return a Spectra object."""
    dic_list, data_list = [], []

    for path in paths_spectra:
        dic, data = ng.fileio.pipe.read(str(path))
        dic_list.append(dic)
        data_list.append(data)
    data = np.concatenate(data_list, axis=0)

    z_values = np.concatenate(
        [np.genfromtxt(path, dtype=None) for path in paths_z_values]
    )

    if exclude_list:
        data, z_values = exclude_planes(data, z_values, exclude_list)

    dic = dic_list[0]
    ucy, ucx = ng.pipe.make_uc(dic, data, dim=1), ng.pipe.make_uc(dic, data, dim=2)

    return Spectra(data, ucx, ucy, z_values)


def exclude_planes(
    data: np.ndarray, z_values: np.ndarray, exclude_list: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude specified planes from data and z_values."""
    exclude_array = np.full_like(z_values, fill_value=False, dtype=np.bool_)
    exclude_array[exclude_list] = True
    return data[~exclude_array], z_values[~exclude_array]


def write_profiles(
    path: Path,
    z_values: np.ndarray,
    cluster: Cluster,
    params: lf.Parameters,
    heights: np.ndarray,
    params_err: dict,
    height_err: np.ndarray,
) -> None:
    """Write profile information to output files."""
    for i, peak in enumerate(cluster.peaks.itertuples()):
        vals, errs = (
            params.valuesdict(),
            {k: v if v is not None else 0.0 for k, v in params_err.items()},
        )

        profile_data = {
            "x_ppm:": f'{vals[f"p{i}_x0"]:10.5f} {errs[f"p{i}_x0"]:10.5f}',
            "y_ppm:": f'{vals[f"p{i}_y0"]:10.5f} {errs[f"p{i}_y0"]:10.5f}',
            "xw_hz:": f'{vals[f"p{i}_x_fwhm"]:10.5f} {errs[f"p{i}_x_fwhm"]:10.5f}',
            "yw_hz:": f'{vals[f"p{i}_y_fwhm"]:10.5f} {errs[f"p{i}_y_fwhm"]:10.5f}',
            "x_eta:": f'{vals[f"p{i}_x_eta"]:10.5f} {errs[f"p{i}_x_eta"]:10.5f}',
            "y_eta:": f'{vals[f"p{i}_y_eta"]:10.5f} {errs[f"p{i}_y_eta"]:10.5f}',
        }

        write_profile(
            path / f"{peak.name}.out",
            peak.name,
            profile_data,
            z_values,
            heights[i],
            height_err[i],
        )


def write_profile(
    filepath: Path,
    name: str,
    profile_data: dict,
    z_values: np.ndarray,
    heights: np.ndarray,
    heights_err: np.ndarray,
) -> None:
    """Write individual profile data to a file."""
    with filepath.open("w") as f:
        f.write(f"# Name: {name}\n")
        for key, value in profile_data.items():
            f.write(f"# {key:<10s} {value}\n")
        f.write("#---------------------------------------------\n")
        f.write(f"# {'Z':>10s}  {'I':>14s}  {'I_err':>14s}\n")
        f.write(
            "\n".join(
                f"  {z!s:>10s}  {ampl:14.6e}  {ampl_e:14.6e}"
                for z, ampl, ampl_e in zip(z_values, heights, heights_err, strict=False)
            )
        )


def get_some_noise(
    spectra: Spectra, noise: np.ndarray, x_pt: np.ndarray, y_pt: np.ndarray
) -> np.ndarray:
    """Retrieve some noise from spectra data."""
    rng = np.random.default_rng()
    rng.shuffle(noise)
    nz_noise, ny_noise, nx_noise = noise.shape

    x_off, y_off = rng.integers(0, nx_noise - 1), rng.integers(0, ny_noise - 1)
    x_noise, y_noise = (x_pt + x_off) % nx_noise, (y_pt + y_off) % ny_noise

    return spectra.data[:, y_noise, x_noise].reshape((nz_noise, x_noise.size)).T


def extract_noise_spectra(
    mc: Sequence[int], spectra: Spectra
) -> tuple[int, np.ndarray]:
    """Extract noise spectra based on monte carlo parameters."""
    x1, x2, y1, y2, n_iter = mc
    x1_pt, x2_pt = sorted((spectra.ucx.i(x1), spectra.ucx.i(x2)))
    y1_pt, y2_pt = sorted((spectra.ucy.i(y1), spectra.ucy.i(y2)))

    noise = spectra.data[:, y1_pt:y2_pt, x1_pt:x2_pt]
    return n_iter, noise


def calc_err_from_mc(
    params_list: Sequence[dict], heights_list: np.ndarray
) -> tuple[dict, np.ndarray]:
    """Calculate errors from monte carlo simulations."""
    params_dict = {}

    for params_mc in params_list:
        for name, param in params_mc.items():
            params_dict.setdefault(name, []).append(param.value)

    params_err = {name: np.std(values) for name, values in params_dict.items()}
    height_err = np.std(heights_list, axis=0, ddof=1)

    return params_err, height_err


def monte_carlo(
    mc: Sequence[int],
    spectra: Spectra,
    cluster: Cluster,
    result: lf.minimizer.MinimizerResult,
    noise: float,
) -> tuple[dict, np.ndarray]:
    """Perform monte carlo simulation for error estimation."""
    n_iter, spectra_noise = extract_noise_spectra(mc, spectra)
    x_pt = np.rint(spectra.ucx.f(cluster.x, "ppm")).astype(int)
    y_pt = np.rint(spectra.ucy.f(cluster.y, "ppm")).astype(int)

    data_sim = simulate_data(result.params, cluster)
    mc_list = []

    for _ in range(int(n_iter)):
        data_noise = get_some_noise(spectra, spectra_noise, x_pt, y_pt)
        data_mc = data_sim + data_noise
        cluster_mc = Cluster(cluster.peaks, cluster.x, cluster.y, data_mc)

        params_mc = lf.minimize(
            residuals,
            result.params,
            args=(cluster_mc, noise),
            method="least_squares",
        ).params

        _, heights = calculate_shape_heights(params_mc, cluster_mc)
        mc_list.append((params_mc, heights))

    params_list, heights_list = zip(*mc_list, strict=False)
    params_err, height_err = calc_err_from_mc(params_list, heights_list)

    return params_err, height_err


def run_fit(clargs, spectra: Spectra, clusters: Sequence[Cluster]) -> dict:
    """Run the fitting process for all clusters."""
    print_fitting()
    shifts = {}

    pdf = PdfPages(clargs.path_output / "clusters.pdf")

    for cluster in clusters:
        print_peaks(cluster.peaks)
        params = create_params(cluster.peaks, spectra, clargs)

        out = lf.minimize(
            residuals,
            params,
            args=(cluster, clargs.noise),
            method="least_squares",
            verbose=2,
        )

        _, heights = calculate_shape_heights(out.params, cluster)

        print_fit_report(out)

        params_err = {param.name: param.stderr for param in out.params.values()}
        height_err = np.full_like(heights, clargs.noise)

        plot_pdf(pdf, spectra, cluster, clargs.contour_level, out)

        shifts.update(
            {
                name: (out.params[f"p{i}_x0"].value, out.params[f"p{i}_y0"].value)
                for i, name in enumerate(cluster.peaks["name"])
            }
        )

        if clargs.mc and int(clargs.mc[4]) > 1:
            params_err, height_err = monte_carlo(
                clargs.mc, spectra, cluster, out, clargs.noise
            )

        write_profiles(
            clargs.path_output,
            spectra.z_values,
            cluster,
            out.params,
            heights,
            params_err,
            height_err,
        )

    pdf.close()

    return shifts


def write_shifts(names: Sequence[str], shifts: dict, file_shifts: Path) -> None:
    """Write the shifts to the output file."""
    with file_shifts.open("w") as f:
        for name in names:
            f.write(f"{name:>15s} {shifts[name][1]:10.5f} {shifts[name][0]:10.5f}\n")


def main() -> None:
    """Run peakfit."""
    print_logo()

    parser = build_parser()
    clargs = parser.parse_args()

    spectra = read_spectra(clargs.path_spectra, clargs.path_z_values, clargs.exclude)

    if clargs.noise is not None and clargs.noise < 0.0:
        clargs.noise = None

    if clargs.noise is None:
        clargs.noise = estimate_noise(spectra.data)
        print_estimated_noise(clargs.noise)

    peaks = read_list(clargs.path_list, spectra)
    clargs.path_output.mkdir(parents=True, exist_ok=True)

    if clargs.contour_level is None:
        clargs.contour_level = 5.0 * clargs.noise

    clusters = cluster_peaks(
        spectra, peaks, clargs.contour_level, clargs.merge_cluster_threshold_hz
    )
    shifts = run_fit(clargs, spectra, clusters)

    export_html(clargs.path_output / "logs.html")
    write_shifts(peaks["name"], shifts, clargs.path_output / "shifts.list")


if __name__ == "__main__":
    main()
