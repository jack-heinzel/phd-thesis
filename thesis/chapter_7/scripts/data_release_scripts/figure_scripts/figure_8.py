import numpy as np
from popsummary.popresult import PopulationResult
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import plot_funcs_bbh_spin as pf

pf.setup()

'''
Setup
'''
# load the popsummary file
result = PopulationResult(fname=pf.DATADIR + "BBHSpin_MagTruncnormIidTiltIsotropicTruncnormSpinSorting.h5")

# rates on grids
chi_gridA, chi_ratesA = result.get_rates_on_grids('chi_A')
chi_gridB, chi_ratesB = result.get_rates_on_grids('chi_B')

# format correctly for pf.plot_traces()
chi_ratesA = np.transpose(chi_ratesA)
chi_ratesB = np.transpose(chi_ratesB)

# more plotting macros
figsize = (pf.FIG_WIDTH/2, 0.7*pf.FIG_WIDTH/2)

'''
Make spin sorted magnitude trace plots
'''

fig, ax = plt.subplots(1,1,figsize=figsize)

pf.plot_traces(ax, chi_gridA, chi_ratesA, color=pf.color_spinA, fill_between=True)
pf.plot_traces(ax, chi_gridB, chi_ratesB, color=pf.color_spinB, fill_between=True)
ax.set_xlabel(r'$\chi$')
ax.set_ylabel(r'$p(\chi)$')
ax.set_ylim(0, )
ax.set_xlim(0.01, 1)

ax.grid(color='silver', alpha=0.5, ls=':')

# legend 
handles = [
    Line2D([], [], color=pf.color_spinA, label=r'$\chi_A$'),
    Line2D([], [], color=pf.color_spinB, label=r'$\chi_B$')
]
ax.legend(handles=handles, loc='upper right')

plt.savefig(pf.figs_folder+'figure_8.pdf', bbox_inches='tight')
plt.close()
