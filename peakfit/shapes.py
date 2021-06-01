import lmfit as lf
import nmrglue.analysis.lineshapes1d as ls


def pvoigt2d(x, y, x0, y0, x_fwhm, y_fwhm, x_eta, y_eta):
    shapex = ls.sim_pvoigt_fwhm(x, x0, x_fwhm, x_eta)
    shapey = ls.sim_pvoigt_fwhm(y, y0, y_fwhm, y_eta)
    return shapex * shapey


def create_params(peaks, ucx, ucy, pvoigt=False, lorenztian=False):

    scalex = abs(ucx.ppm(ucx.f(10, "hz")))
    scaley = abs(ucy.ppm(ucy.f(10, "hz")))

    params = lf.Parameters()

    for index, (_, x0, _, y0, _) in enumerate(peaks):
        pre = f"p{index}_"
        x0_name = "".join([pre, "x0"])
        y0_name = "".join([pre, "y0"])
        x_fwhm_name = "".join([pre, "x_fwhm"])
        y_fwhm_name = "".join([pre, "y_fwhm"])
        x_eta_name = "".join([pre, "x_eta"])
        y_eta_name = "".join([pre, "y_eta"])

        params.add(x0_name, value=x0, min=x0 - scalex, max=x0 + scalex, vary=False)
        params.add(y0_name, value=y0, min=y0 - scaley, max=y0 + scaley, vary=False)

        params.add(x_fwhm_name, value=20.0, min=0.1, max=200.0)
        params.add(y_fwhm_name, value=20.0, min=0.1, max=200.0)

        # By default, the shape is gaussian (eta = 0.0)
        if pvoigt:
            params.add(x_eta_name, value=0.5, min=-1.0, max=+1.0, vary=False)
            params.add(y_eta_name, value=0.5, min=-1.0, max=+1.0, vary=False)
        elif lorenztian:
            params.add(x_eta_name, value=1.0, min=-1.0, max=+1.0, vary=False)
            params.add(y_eta_name, value=1.0, min=-1.0, max=+1.0, vary=False)
        else:
            params.add(x_eta_name, value=0.0, min=-1.0, max=+1.0, vary=False)
            params.add(y_eta_name, value=0.0, min=-1.0, max=+1.0, vary=False)

    return params
