from popsummary.popresult import PopulationResult
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import plot_funcs_bbh_mass as pf

pf.setup()    

iso = pf.GetBSplineMacroData(fname = '../data_release/BBHMassSpinRedshift_BSplineIsopeakIID.h5', iid = True, isopeak = True)
tom = PopulationResult(fname = '../data_release/BBHMass_VaryingBetaQs_DominantMode.h5')

tomx, tompx = tom.get_rates_on_grids(grid_key='dP_dq_lowMassPeak')
tomx = tomx[0]

tomx2, tompx2 = tom.get_rates_on_grids(grid_key='dP_dq_highMassPeak')
tomx2 = tomx2[0]

tomx3, tompx3 = tom.get_rates_on_grids(grid_key='dP_dq_powerLaw')
tomx3 = tomx3[0]

def norm(px, x):
    norm = trapezoid(px, x, axis = 1)
    return px / norm.reshape(-1,1)


colors = ['#FE6100', '#785EF0']

fig, axes = plt.subplots(nrows = 2, figsize=(5.7,7.5))

color2 = 'k'
color1 = '#DC267F'
ax = axes[1]
pf.setup_mass_ratio_plot(ax, grid_kwargs={'ls': ':'})

pf.plot_90CI(ax, iso.pdfs['peak_mass_ratio'], norm(iso.pdfs['peak_mass_ratio_pdfs'], iso.pdfs['peak_mass_ratio']), color = pf.darken('#648FFF'), label = r'$\mathrm{Peak}$, ${\sim}10\, \mathrm{M}_\odot$', median = False, fill = False, secondary_ls = '-')
pf.plot_90CI(ax, iso.pdfs['peak_mass_ratio'], norm(iso.pdfs['continuum_mass_ratio_pdfs'], iso.pdfs['continuum_mass_ratio']), color = pf.darken('#648FFF'), label = r'$\mathrm{Continuum}$', median = False, fill = True)

pf.plot_90CI(ax, tomx, tompx, color = pf.darken('#FE6100'), fill = False, secondary_ls='-', median = False)

ax.set_ylim(1e-3,5e1)
ax.set_ylabel(r'$p(q)$')
ax.legend(fontsize = 11, title = r'$\textsc{Isolated Peak}$')


color1 = '#FE6100'#'#785EF0'
color2 = '#FE6100'
ax = axes[0]
pf.setup_mass_ratio_plot(ax, grid_kwargs={'ls': ':'})
pf.plot_90CI(ax, tomx, tompx, color = pf.darken(color1), secondary_ls='-', label = r'$\mathrm{Peak} \, 1$, ${\sim} 10 \, \mathrm{M}_\odot$', median = False, fill = False)
pf.plot_90CI(ax, tomx2, tompx2, color = pf.darken(color1), secondary_ls='--', median = False, fill = False, label = r'$\mathrm{Peak} \, 2$, ${\sim} 35 \, \mathrm{M}_\odot$')
pf.plot_90CI(ax, tomx3, tompx3, color = color1, label = r'$\mathrm{Broken} \, \mathrm{Power} \, \mathrm{Law}$', secondary_ls='-', median = False, fill_alpha = 0.5)
ax.set_ylim(1e-3,5e1)
ax.set_ylabel(r'$p(q)$')
ax.legend(title=r"$\textsc{Extended BP2P}$", fontsize = 11)



plt.tight_layout()
plt.savefig('../figures/figure_6.pdf')
