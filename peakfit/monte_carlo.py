import itertools


def get_noise(spectra, x_box_min, x_box_max, y_box_min, y_box_max):
    if x_box_min > x_box_max:
        x_box_min, x_box_max = x_box_max, x_box_min

    if y_box_min > y_box_max:
        y_box_min, y_box_max = y_box_max, y_box_min

    x_box_min = max([x_box_min, 0])
    x_box_max = min([x_box_max, spectra.shape[2]])
    y_box_min = max([y_box_min, 0])
    y_box_max = min([y_box_max, spectra.shape[1]])

    return np.std(spectra[:, y_box_min:y_box_max, x_box_min:x_box_max], ddof=1)


def get_noise_grid(x_box_min, x_box_max, y_box_min, y_box_max, x_grid, y_grid):
    if x_box_min > x_box_max:
        x_box_min, x_box_max = x_box_max, x_box_min

    if y_box_min > y_box_max:
        y_box_min, y_box_max = y_box_max, y_box_min

    x_grid_min = min(x_grid)
    x_grid_max = max(x_grid)

    y_grid_min = min(y_grid)
    y_grid_max = max(y_grid)

    x_box_max -= x_grid_max - x_grid_min
    y_box_max -= y_grid_max - y_grid_min

    grid_noise = list(itertools.product(range(x_box_min, x_box_max), range(y_box_min, y_box_max)))

    return grid_noise
