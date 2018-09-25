import itertools


def get_noise_grid(bounds_x, bounds_y, x_grid, y_grid):

    range_x = max(x_grid) - min(x_grid)
    range_y = max(y_grid) - min(y_grid)

    bounds_x[1] -= range_x
    bounds_y[1] -= range_y

    grid_noise = list(itertools.product(range(*bounds_x), range(*bounds_y)))

    return grid_noise
