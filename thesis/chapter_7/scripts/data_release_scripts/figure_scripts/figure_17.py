import os
import numpy as np
from popsummary.popresult import PopulationResult
from matplotlib import pyplot as plt
import plot_funcs_bbh_spin as pf
import seaborn as sns

pf.setup()

'''
Setup
'''

# Load inputs
fpaths = ["../data_release/BBHSpin_MagTruncnormIndTiltIsotropicTruncnormNnd.h5",
          "../data_release/BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid.h5"]

popsummary_results = {}
for fpath in fpaths: 
    key = os.path.basename(fpath).split(".h5")[0]
    popsummary_results[key] = PopulationResult(fname=fpath)
    
# Set up figure and aesthetics
figsize = (1.5*pf.FIG_WIDTH, 1.5*0.6*pf.FIG_WIDTH/2)

color1 = '#3484eb'  # blue
color2 = '#eb4034'  # red


'''
Make cornerplot 
'''

i = 'BBHSpin_MagTruncnormIndTiltIsotropicTruncnormNnd'
hp_list1 = ['mu_chi_1', 'sigma_chi_1', 'mu_1', 'sigma_1']
hyperposterior1 = popsummary_results[i].get_hyperparameter_samples(hyperparameters=hp_list1)
hp_dict1 = {k:v for k,v in zip(hp_list1, hyperposterior1.T)}
hp_list2 = ['mu_chi_2', 'sigma_chi_2', 'mu_2', 'sigma_2']
hyperposterior2 = popsummary_results[i].get_hyperparameter_samples(hyperparameters=hp_list2)
hp_dict2 = {k:v for k,v in zip(hp_list2, hyperposterior2.T)}

i = 'BBHSpin_MagTruncnormIidTiltIsotropicTruncnormNid'
hp_list_id = ['mu_chi', 'sigma_chi', 'mu_spin', 'sigma_spin']
hyperposterior_id = popsummary_results[i].get_hyperparameter_samples(hyperparameters=hp_list_id)
hp_dict_id = {k:v for k,v in zip(hp_list_id, hyperposterior_id.T)}

hist_kws = {
    '1':dict(density=True, color=color1,histtype='step', lw=2, zorder=1),
    '2':dict(density=True, color=color2, histtype='step', lw=2, zorder=2),
    'identical':dict(density=True, color='gray', alpha=0.5, zorder=0)
}
nbins=30
limits_dict = {'mu_chi':[0,0.5], 'sigma_chi':[0,0.8], 'mu':[-1,1], 'sigma':[0,4]}
params = list(limits_dict.keys())

fig, axes = plt.subplots(4,4, figsize=(pf.FIG_WIDTH,pf.FIG_WIDTH))

## 1D HISTS

# mu_chi, sigma_chi, mu_cos_theta, sigma_cos_theta

i=0; k='mu_chi'
axes[i][i].hist(hp_dict1[f'{k}_1'], **hist_kws['1'], bins=np.linspace(*limits_dict[k], nbins), label='BH 1')
axes[i][i].hist(hp_dict2[f'{k}_2'], **hist_kws['2'], bins=np.linspace(*limits_dict[k], nbins), label='BH 2')
for ax in axes[:,i]: 
    ax.set_xlim(*limits_dict[k])

for _i, k in enumerate(params[1:]): 
    i = _i + 1
    axes[i][i].hist(hp_dict1[f'{k}_1'], **hist_kws['1'], bins=np.linspace(*limits_dict[k], nbins))
    axes[i][i].hist(hp_dict2[f'{k}_2'], **hist_kws['2'], bins=np.linspace(*limits_dict[k], nbins))
    for ax in axes[:,i]: 
        ax.set_xlim(*limits_dict[k])
        
# add identical
for i,k in enumerate(hp_dict_id.keys()):
    axes[i][i].hist(hp_dict_id[k], **hist_kws['identical'], bins=np.linspace(*limits_dict[list(limits_dict.keys())[i]], nbins), label='identical')
    
axes[0][0].legend(bbox_to_anchor=(2,1), frameon=False)


## 2D HISTS
# identical
for i, ki in enumerate(hp_dict_id.keys()):
    for j, kj in enumerate(hp_dict_id.keys()):
        if i > j: 
            sns.kdeplot(data=hp_dict_id, y=ki, x=kj, ax=axes[i][j], color='gray', levels=[0.5, 0.9], fill=True)
# nonidentical
for i, ki in enumerate(params): 
    for j, kj in enumerate(params): 
        if i > j: 
            sns.kdeplot(data=hp_dict1, y=f"{ki}_1", x=f"{kj}_1", ax=axes[i][j], color=color1, levels=[0.5, 0.9])
            sns.kdeplot(data=hp_dict2, y=f"{ki}_2", x=f"{kj}_2", ax=axes[i][j], color=color2, levels=[0.5, 0.9])
    
# axes labeling
for i in range(4): 
    for j in range(4): 
        if i < j: 
            axes[i][j].set_visible(False)
            
        if i!=3: 
            axes[i][j].set_xticklabels('')

        if j!=0: 
            axes[i][j].set_yticklabels('')
            axes[i][j].set_ylabel('')
            
        axes[i][j].grid(color='silver', alpha=0.5, ls=':')      
        
labels = [r'$\mu_\chi$', r'$\sigma_\chi$', r'$\mu_t$', r'$\sigma_t$']
for ax,l in zip(axes[3],labels): 
    ax.set_xlabel(l)
for ax,l in zip(axes[:,0],labels): 
    ax.set_ylabel(l)
axes[0][0].set_yticklabels('')
axes[0][0].set_ylabel('')

plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.savefig(pf.figs_folder+'figure_17.pdf', bbox_inches='tight')
plt.close()
