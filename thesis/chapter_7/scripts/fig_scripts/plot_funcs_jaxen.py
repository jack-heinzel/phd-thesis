
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import h5py
from bilby.core.result import read_in_result

def setup(): 

    from matplotlib import rc, rcParams

    rc_params = {
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "font.size": 12,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "text.usetex": True,
        "savefig.dpi": 300,
    }
    rcParams.update(rc_params)
    rc("text", usetex=True)
    rc("axes", linewidth=0.5)
    rcParams["xtick.major.pad"] = "6"
    rcParams["ytick.major.pad"] = "6"



def get_params(dataset, param, rate = True):
    param = [param] if not isinstance(param, list) else param
    for p in param:
        dat = dataset.get_rates_on_grids(p)
        x = dat[0][0]
        pdf = dat[1]
    if rate:
        lamb = dataset.get_hyperparameter_samples(hyperparameters = 'lamb')
        pdf = pdf * (1 + 0.2)**lamb.reshape(-1,1)
    return x, pdf



def plot_90CI(ax, x, pdf, fill = True, label = '', color = 'r', fill_alpha = 0.3, lw = 2, ls = '-', secondary_ls = '--',
        plot_kwargs=dict(rasterized=True), fill_kwargs=dict(rasterized=True), median=True, mean=False):
    med = np.median(pdf, axis=0) if median else np.mean(pdf, axis=0)
    low = np.percentile(pdf, 5, axis=0)
    high = np.percentile(pdf, 95, axis=0)

    if median:
        ax.plot(x, med, **plot_kwargs, color = color, label = label, lw = lw, ls = ls)
        label = None
    if fill:
        ax.fill_between(x, low, high, color = color, alpha = fill_alpha, **fill_kwargs, label = label)
        label = None
    else:
        ax.plot(x, low, **plot_kwargs, color = color, lw = lw, ls = secondary_ls, label = label)
        ax.plot(x, high, **plot_kwargs, color = color, lw = lw, ls = secondary_ls)


def get_03b_plp_ppds(path, mass_1 = True, mass_ratio = False):
    o3b_result = read_in_result(path + '/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json')
    o3b_hyper = o3b_result.posterior.copy()
    rate_o2 = (1 + 0.2)**np.mean(o3b_hyper['lamb'].values)
    with h5py.File(path + '/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_mass_data.h5','r') as f:
        ppd = f['ppd'][:] * rate_o2
        mlines = f['lines']['mass_1'][:] * rate_o2
        qlines = f['lines']['mass_ratio'][:] * rate_o2
    m1 = np.linspace(2,100,1000)
    q = np.linspace(0.1,1,500)
    mu = trapezoid(ppd, q, axis = 0) if mass_1 else trapezoid(ppd, m1, axis = 1)
    lines = mlines if mass_1 else qlines
    low = np.percentile(lines, 5, axis = 0)
    hi = np.percentile(lines, 95, axis = 0)
    x = m1 if mass_1 else q
    return x, low, mu, hi



def setup_mass_plot(ax, xrange=(2,100), yrange=(1e-3,20), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(r"$m_1 \left[ \mathrm{M}_\odot \right]$", **label_kwargs)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}m_1 \left[ \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1} \, \mathrm{M}_\odot^{-1} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)


def setup_mass_ratio_plot(ax, xrange=(0,1), yrange=(4e-2,3e2), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(r"$q$", **label_kwargs)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}q \left[ \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)


def darken(hex_color, factor = 0.8):
    """Darkens a given hex color by a specified factor.

    Args:
        hex_color: The hex color string (e.g., "#FF0000").
        factor: A float between 0 and 1, where 0 is fully black and 1 is no change.

    Returns:
        A darkened hex color string.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r * factor)
    g = int(g * factor)
    b = int(b * factor)

    return f"#{r:02x}{g:02x}{b:02x}"
