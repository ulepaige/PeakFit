import lmfit as lf
import nmrglue.analysis.lineshapes1d as ls


def pvoigt2d(x, y, x0, y0, x_fwhm, y_fwhm, x_eta, y_eta):
    return ls.sim_pvoigt_fwhm(x, x0, x_fwhm, x_eta) * ls.sim_pvoigt_fwhm(y, y0, y_fwhm, y_eta)


def create_params(peaks):
    params = lf.Parameters()

    for index, peak in enumerate(peaks):
        prefix = 'p{}_'.format(index)
        x0, y0 = peak[1:3]

        x0_name = ''.join([prefix, 'x0'])
        y0_name = ''.join([prefix, 'y0'])
        x_fwhm_name = ''.join([prefix, 'x_fwhm'])
        y_fwhm_name = ''.join([prefix, 'y_fwhm'])
        x_eta_name = ''.join([prefix, 'x_eta'])
        y_eta_name = ''.join([prefix, 'y_eta'])

        params.add(x0_name, value=x0)
        params.add(y0_name, value=y0)
        params.add(x_fwhm_name, value=3.0, min=0.001, max=10.0)
        params.add(y_fwhm_name, value=3.0, min=0.001, max=10.0)
        params.add(x_eta_name, value=0.0, min=-1.0, max=+1.0, vary=False)
        params.add(y_eta_name, value=0.0, min=-1.0, max=+1.0, vary=False)

    return params
