import popsummary
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.colors import LogNorm

plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=13)

fig, axes = plt.subplots(ncols=4, figsize=(10,4), width_ratios=[0.075,1,1,0.075])

uncmin = 0.5
uncmax = 50

majors = [1,3,10,30,100]
major_names = ['1','3','10','30','100']
xticks=np.log(np.concatenate((np.arange(1,10), np.arange(10,110,10))))

DATADIR = "../data_release/"
pdb_file = DATADIR + 'AllCBC_FullPop.h5'
pdb_result = popsummary.popresult.PopulationResult(fname=pdb_file)

bgp_file = DATADIR + 'AllCBC_FullPopBGP.h5'
bgp_result = popsummary.popresult.PopulationResult(fname=bgp_file)

(bgp_m1, bgp_m2), bgp_R = bgp_result.get_rates_on_grids('ppd_primary_and_secondary_mass')
(m1, m2), R = pdb_result.get_rates_on_grids('primary_mass_secondary_mass_joint_median')

max_ind = np.digitize(180, m1) + 1
R = R[:max_ind, :max_ind]

(_, _), pdb_U = pdb_result.get_rates_on_grids('primary_mass_secondary_mass_joint_uncertainty')
pdb_U = np.nan_to_num(pdb_U)
pdb_U = pdb_U + pdb_U.T - np.diag(np.diag(pdb_U))
pdb_U = pdb_U[:max_ind, :max_ind]

(_, _), bgp_U = bgp_result.get_rates_on_grids('uncert_ppd_primary_and_secondary_mass')

pdb_ax = axes[1]
bgp_ax = axes[2]

m1 = m1[:max_ind]
m2 = m2[:max_ind]

R *= np.outer(m1, m2)
impdb = pdb_ax.imshow(
    R, 
    cmap='Blues', 
    norm=LogNorm(vmin=1e-3, vmax=1e3), 
    origin='lower', 
    extent=(0,np.log(180),0,np.log(180))
    )
imbgp = bgp_ax.imshow(
    bgp_R, 
    cmap='Blues', 
    norm=LogNorm(vmin=1e-3, vmax=1e3), 
    origin='lower', 
    extent=(0,np.log(180),0,np.log(180))
    )

unpdb = pdb_ax.imshow(
    pdb_U.T,
    cmap='Reds', 
    norm=LogNorm(vmin=uncmin, vmax=uncmax), 
    origin='lower', 
    extent=(0,np.log(180),0,np.log(180))
)
unbgp = bgp_ax.imshow(
    bgp_U.T,
    cmap='Reds', 
    norm=LogNorm(vmin=uncmin, vmax=uncmax), 
    origin='lower', 
    extent=(0,np.log(180),0,np.log(180))
)
for il, iu, a in zip([impdb, imbgp], [unpdb, unbgp], [pdb_ax, bgp_ax]):
    a.set_xticks(xticks, minor=True)
    a.set_yticks(xticks, minor=True)
    a.set_xticks(np.log(np.array(majors)), major_names, minor=False)
    a.set_yticks(np.log(np.array(majors)), major_names, minor=False)
    il.set_clip_path(
        Polygon(
            np.array([[np.log(1), np.log(1)], [np.log(180), np.log(1)], [np.log(180), np.log(180)]]),
            closed=True, transform=a.transData
            )
        )
    iu.set_clip_path(
        Polygon(
            np.array([[np.log(1), np.log(1)], [np.log(1), np.log(180)], [np.log(180), np.log(180)]]),
            closed=True, transform=a.transData
            )
        )
    a.set_xlabel('$m_1$ [$M_\odot$]')
    a.grid(ls = ':', alpha = 0.5, lw = 1, color = 'k')

vmin, vmax = 1e-3, 1e3
nOrder = int(np.log10(vmax/vmin) + 0.01)
tick_spacing = 2

unc_tick_spacing = 1
nOrderunc = int(np.log10(uncmax+0.01)) - int(np.log10(uncmin + 0.01))

cbarlower = fig.colorbar(il, cax=axes[3], ticks=np.logspace(np.log10(vmin), np.log10(vmax), nOrder+1))
cbarupper = fig.colorbar(iu, cax=axes[0])

tick_labels = []
for i, x in enumerate(np.linspace(np.log10(vmin), np.log10(vmax), nOrder+1, dtype=int)):
    if i % tick_spacing == 0:
        tick_labels.append(f'$10^{{{x}}}$')
    else:
        tick_labels.append('')

unc_tick_labels = []
for i, x in enumerate(np.linspace(int(np.log10(uncmin+0.01))+1, int(np.log10(uncmax+0.01))+1, nOrderunc+1, dtype=int)):
    if i % unc_tick_spacing == 0:
        unc_tick_labels.append(f'$10^{{{x}}}$')
    else:
        unc_tick_labels.append('')

cbarlower.ax.set_yticklabels(tick_labels)
cbarlower.set_label('$\\frac{\mathrm{d}\mathcal{R}}{\mathrm{d}(\ln m_1)\mathrm{d}(\ln m_2)}$ [Gpc${}^{-3}$ yr${}^{-1}$]', rotation=90)

cbarupper.set_label('Fractional uncertainty', rotation=90)

cbarlower.ax.yaxis.set_label_position('left')
cbarupper.ax.yaxis.set_label_position('left')
pdb_ax.set_title(r'\textsc{FullPop}-4.0')
pdb_ax.set_ylabel('$m_2$ [$M_\odot$]')
bgp_ax.set_title('BGP')

plt.tight_layout(pad=0., w_pad=0, h_pad=1.0)

for ax in [axes[0], axes[3]]:
    pos = ax.get_position()
    yscale = pos.y1-pos.y0
    w = 0.051
    ax.set_position([pos.x0*0.95, pos.y0 + yscale*w, pos.x1-pos.x0, yscale*(1 - 2*w)])

plt.savefig('../figures/figure_1.pdf', bbox_inches='tight', dpi=200)
