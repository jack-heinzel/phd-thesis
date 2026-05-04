# import numpy as np
from matplotlib import rc, rcParams

rc_params = {
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.usetex": True,
    "savefig.dpi": 300,
}

rcParams.update(rc_params)

rc("text", usetex=True)
rc("axes", linewidth=0.5)

rcParams["xtick.major.pad"] = "6"
rcParams["ytick.major.pad"] = "6"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import seaborn

seaborn.set_palette("colorblind")

o4a_ns_pop_dir = (
    "/home/aditya.vijaykumar/work/O4/O4a-astro/o4a-ns-pop/dat/NeutronStarMass/"
)

gaia_mean_masses = [
    1.33,
    1.31,
    1.323,
    1.473,
    1.28,
    1.291,
    1.34,
    1.362,
    1.39,
    1.258,
    1.396,
    1.7,
    1.38,
    1.362,
    1.898,
    1.52,
    1.396,
    1.443,
    1.604,
    1.401,
    1.48,
]


which_models = ["peakcut_m1m2", "power_m1m2"]
which_set = ["setA"]


for ws in range(len(which_set)):

    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3))

    med = True

    lw = 1.5
    fontsize = 12
    alpha = 0.2

    if med:
        cent = [2, 6, 10, 14, 18, 22]
    else:
        cent = [1, 5, 9, 13, 17, 21]

    x_lims = [0.9, 3]
    y_lims = [0.1, 7]

    for i in range(2):

        ppd_vals_O3 = np.array(
            pd.read_csv(
                o4a_ns_pop_dir
                + "O3b_rerun/"
                + "{}-{}-semianalyticvt/{}_ppd.csv".format(
                    which_set[ws], which_models[i], which_models[i]
                )
            )
        )
        ax[i].semilogy(
            ppd_vals_O3[:, 0],
            ppd_vals_O3[:, cent[0]],
            c=sns.color_palette()[0],
            label="GWTC-3",
            rasterized=True,
        )
        ax[i].fill_between(
            ppd_vals_O3[:, 0],
            ppd_vals_O3[:, 3],
            ppd_vals_O3[:, 4],
            color=sns.color_palette()[0],
            alpha=alpha,
            lw=lw,
            rasterized=True,
        )

        ppd_vals_O4 = np.array(
            pd.read_csv(
                o4a_ns_pop_dir
                + "O4a_O1O2O3VT/"
                + "{}-{}-semianalyticvt/{}_ppd.csv".format(
                    which_set[ws], which_models[i], which_models[i]
                )
            )
        )
        ax[i].semilogy(
            ppd_vals_O4[:, 0],
            ppd_vals_O4[:, cent[0]],
            c=sns.color_palette()[1],
            label="GWTC-4",
            rasterized=True,
        )
        ax[i].fill_between(
            ppd_vals_O4[:, 0],
            ppd_vals_O4[:, 3],
            ppd_vals_O4[:, 4],
            color=sns.color_palette()[1],
            alpha=alpha,
            lw=lw,
            rasterized=True,
        )

        ax[i].tick_params(labelsize=fontsize)
        ax[i].set_xlim(x_lims[0], x_lims[1])
        ax[i].set_ylim(y_lims[0], y_lims[1])
        ax[i].set_xlabel(r"$m_\mathrm{NS}\, [M_{\odot}]$", fontsize=fontsize)
        ax[i].set_ylabel(r"$p_{\Lambda}(m_\mathrm{NS})$", fontsize=fontsize + 2)
        ax[i].grid(ls="dotted")
        ax[i].legend(fontsize=fontsize)
        # ax[i].vlines(
        #     gaia_mean_masses,
        #     y_lims[0],
        #     1.5 * y_lims[0],
        #     colors=sns.color_palette()[2],
        #     lw=0.5,
        #     rasterized=True,
        # )

    fig.tight_layout()
    fig.savefig("../../figures/ppd_ns_mass.pdf", dpi=300)
