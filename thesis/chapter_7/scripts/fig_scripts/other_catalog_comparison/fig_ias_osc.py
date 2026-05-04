"""
Run this using a conda environment with pyarrow installed
python3 fig_ias_osc.py 

Part of the data release: FIXME

Authors: Shio Sakon on behalf of the LIGO Scientific Collaboration, Virgo Collaboration and KAGRA Collaboration, originally created by Stephen Fairhurst

This software is provided under license: Creative Commons Attribution 4.0
International (https://creativecommons.org/licenses/by/4.0/legalcode) and is provided as-is.

This makes Figure FIXME in FIXME 
"""

#!/urs/bin/env python3

import sys
import json
from os.path import exists
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import rcParams
from scipy.ndimage.filters import gaussian_filter
from matplotlib.patches import Rectangle
from tqdm.notebook import tqdm
from scipy.special import erf
from matplotlib import cm
from scipy.stats import gaussian_kde
from astropy.cosmology import Planck15, z_at_value
import astropy.units as u

from utils import mpl_utils


rcParams["text.usetex"] = True
rcParams["font.serif"] = "Computer Modern"
rcParams["font.family"] = "Serif"
rcParams["xtick.labelsize"]=14
rcParams["ytick.labelsize"]=14
rcParams["xtick.direction"]="in"
rcParams["ytick.direction"]="in"
rcParams["legend.fontsize"]=15
rcParams["axes.labelsize"]=16
rcParams["axes.grid"] = True
rcParams["grid.color"] = 'black'
rcParams["grid.linewidth"] = 1.
rcParams["grid.alpha"] = 0.6

# details of other populations
event_dir = 'external_catalog_data/'

print('Loading events from IAS catalogs')
sys.path.append(event_dir)
 
# IAS samples
ias_events = ['GW151216',
              'GW170121',
              'GW170202',
              'GW170304',
              'GW170403',
              'GW170425',
              'GW170727']
for event in (ias_events + ['load_samples.py']):
    #print( event )
    if (event == 'load_samples.py'):
        fn = event
    else:
        fn = '%s.npy' % event
    fp = "%s/%s" % (event_dir, fn)
    if exists(fp):
        print(fn,'exists')
    else:
        print(fn,"doesn't exist; please download")
        sys.exit(1)

import load_samples

ias_samples = load_samples.load_samples(ias_events, event_dir)

# calculate required quantities, restrict to 1000 random samples
ias_sample_dict = {}
for (event, post_samples) in ias_samples.items():
    random_inds = np.random.choice(np.arange(len(post_samples['mchirp'])),size=1000,replace=False)
    DL = post_samples['DL'][random_inds]
    z = np.array([z_at_value(Planck15.luminosity_distance, DLs * u.Mpc) for DLs in DL])
    Mc = post_samples['mchirp'][random_inds]/(1 + z)
    eta = post_samples['eta'][random_inds]
    Mt = Mc/eta**(3./5)
    m1 = 0.5 * Mt * (1.0 + (1.0 - 4.0 * eta)**0.5)
    m2 = 0.5 * Mt * (1.0 - (1.0 - 4.0 * eta)**0.5)
    s1z = post_samples['s1z'][random_inds]
    s2z = post_samples['s2z'][random_inds]
    Xeff = (s1z * m1 + s2z * m2) / Mt
    ias_sample_dict[event] = {'m1':m1,'m2':m2,'z':z,'Xeff':Xeff,'Mc':Mc}

## IAS O3a samples
ias_events_O3a = ['GW190704_104834', 'GW190707_083226', 'GW190711_030756', 'GW190718_160159', 'GW190814_192009', 'GW190818_232544', 'GW190821_124821', 'GW190906_054335', 'GW190910_012619', 'GW190920_113516']

import pandas

for event in ias_events_O3a:
    fn = '%s_posterior_samples.feather' % event
    fp = "%s/%s" % (event_dir, fn)
    df = pandas.read_feather(fp)
    random_inds = np.random.choice(np.arange(len(df['m1'])),size=1000,replace=False)
    m1 = df['m1_source'][:][random_inds]
    m2 = df['m2_source'][:][random_inds]
    z = df['z'][:][random_inds]
    Xeff = df['chieff'][:][random_inds]
    Mc = df['mchirp_source'][:][random_inds]
    ias_sample_dict[event] = {'m1':m1,'m2':m2,'z':z,'Xeff':Xeff,'Mc':Mc}
    #print("loaded data for %s" % event)

## IAS O3b samples
ias_events_O3b = ['GW191117_023843', 'GW191228_085854', 'GW200109_195634', 'GW200210_100022', 'GW200225_075134', 'GW200316_235947'] 

for event in ias_events_O3b:
    fn = '%s_posterior_samples.feather' % event
    fp = "%s/%s" % (event_dir, fn)
    df = pandas.read_feather(fp)
    random_inds = np.random.choice(np.arange(len(df['m1'])),size=1000,replace=False)
    DL = df['d_luminosity'][:][random_inds]
    z = np.array([z_at_value(Planck15.luminosity_distance, DLs * u.Mpc) for DLs in DL])
    Mc = df['mchirp'][:][random_inds]/(1+z)
    m1 = df['m1'][:][random_inds]/(1+z)
    m2 = df['m2'][:][random_inds]/(1+z)
    Xeff = df['chieff'][:][random_inds]
    ias_sample_dict[event] = {'m1':m1,'m2':m2,'z':z,'Xeff':Xeff,'Mc':Mc}
    #print("loaded data for %s" % event)

# OGC samples
print('Loading events from OGC catalogs')
#ogc_events = ['GW170121_212536', 'GW170304_163753', 'GW170727_010430', 'GW190725_174728', 'GW190916_200658', 'GW190925_232845', 'GW190926_050336', 'GW191224_043228', 'GW200106_134123', 'GW200129_114245', 'GW200210_005122', 'GW200214_223307', 'GW200305_084739', 'GW200318_191337']
# only include events that are not in IAS and not in GWTC
ogc_events = ['GW191224_043228', 'GW200106_134123', 'GW200129_114245', 'GW200210_005122', 'GW200214_223307', 'GW200305_084739', 'GW200318_191337']
ogc_sample_dict = {}
    
for event in ogc_events:
    fn = "%s-PYCBC-POSTERIOR-IMRPhenomXPHM.hdf" % event
    fp = "%s/%s" % (event_dir, fn)
    if exists(fp):
        print('%s exists' % fn)
    else:
        print("%s doesn't exist; please download" % fn)
        sys.exit(1)
                
    f = h5py.File(fp, 'r')               
    post_samples = f['samples']
    
    random_inds = np.random.choice(np.arange(len(post_samples['srcmass1'])),size=1000,replace=False)
    m1 = post_samples['srcmass1'][:][random_inds]
    m2 = post_samples['srcmass2'][:][random_inds]
    Xeff = post_samples['chi_eff'][:][random_inds]
    z = post_samples['redshift'][:][random_inds]
    Mc = post_samples['srcmchirp'][:][random_inds]
    ogc_sample_dict[event] = {'m1':m1,'m2':m2,'z':z,'Xeff':Xeff,'Mc':Mc}
    #print("loaded data for %s" % event)
    
"""
# LVK samples:
print('Loading events from LVK catalogs')
# FIXME sampleFile = "postproc/O4a_vs_other_catalog_comparison_data.json"
sampleFile = "postproc/O3_vs_other_blob_comparison_data.json"
with open(sampleFile,'r') as jf:
    lvk_sample_dict = json.load(jf)

# Convert data to numpy arrays
for post_samples in lvk_sample_dict.values():
    for prop in post_samples.keys():
        post_samples[prop] = np.array(post_samples[prop])
       
        
sampleDicts = {'LVK':lvk_sample_dict,
               'IAS':ias_sample_dict,
               'OGC':ogc_sample_dict
              }
colors = {'OGC':'#33a02c',
          'IAS':'#e31a1c',
          'LVK':'#08519c'}
"""

sampleDicts = {
               'IAS':ias_sample_dict,
               'OGC':ogc_sample_dict
              }
# Use CVD-friendly colors 
colors = {'OGC':'#1a80bb',
          'IAS':'#ea801c'}

# Load population fit
# FIXME: use O4a population models 
tf = "analyses/PowerLawPeak/o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json"
with open(tf) as jf:
    data = json.load(jf)
    
posterior_samples = data['posterior']['content']

# Read out parameters
mass_alphas = np.array(posterior_samples['alpha'])
mass_betas = np.array(posterior_samples['beta'])
delta_ms = np.array(posterior_samples['delta_m'])
mmins = np.array(posterior_samples['mmin'])
mmaxs = np.array(posterior_samples['mmax'])
lams = np.array(posterior_samples['lam'])
mu_ms = np.array(posterior_samples['mpp'])
sig_ms = np.array(posterior_samples['sigpp'])
mu_x = np.array(posterior_samples['mu_chi'])
sigma2_x = np.array(posterior_samples['sigma_chi'])
zetas = np.array(posterior_samples['xi_spin'])
sig_ts = np.array(posterior_samples['sigma_spin'])
kappas = np.array(posterior_samples['lamb'])

# Convert spin mean/variances to beta distribution alpha and beta parameters
spin_alphas = ((1.-mu_x)/sigma2_x - 1./mu_x)*mu_x*mu_x
spin_betas = spin_alphas*(1./mu_x-1.)

# Now we need to load injections to calibrate our selection effects
# FIXME: change to O4a injection files
mockDetections = h5py.File('injections/LIGO-T2100113-v12/endo3_bbhpop-LIGO-T2100113-v12.hdf5','r')
ifar_1 = mockDetections['injections']['ifar_gstlal'][()]
ifar_2 = mockDetections['injections']['ifar_pycbc_bbh'][()]
ifar_3 = mockDetections['injections']['ifar_pycbc_hyperbank'][()]
detected = (ifar_1>1) + (ifar_2>1) + (ifar_3>1)
m1_det = mockDetections['injections']['mass1_source'][()][detected]
m2_det = mockDetections['injections']['mass2_source'][()][detected]
s1x_det = mockDetections['injections']['spin1x'][()][detected]
s1y_det = mockDetections['injections']['spin1y'][()][detected]
s1z_det = mockDetections['injections']['spin1z'][()][detected]
s2x_det = mockDetections['injections']['spin2x'][()][detected]
s2y_det = mockDetections['injections']['spin2y'][()][detected]
s2z_det = mockDetections['injections']['spin2z'][()][detected]
z_det = mockDetections['injections']['redshift'][()][detected]

# Read injection weights from injection file
# Note that injected distribution is *flat* in spin magnitude and spin orientation
m1_m2_z_sampling = (mockDetections['injections']['mass1_source_mass2_source_sampling_pdf'][()]*mockDetections['injections']['redshift_sampling_pdf'][()])[detected]
pop_reweight = 1./m1_m2_z_sampling

a1_det = np.sqrt(s1x_det**2+s1y_det**2+s1z_det**2)
a2_det = np.sqrt(s2x_det**2+s2y_det**2+s2z_det**2)
cost1_det = s1z_det/a1_det
cost2_det = s2z_det/a2_det
dVdz_det = 4.*np.pi*Planck15.differential_comoving_volume(z_det).to(u.Gpc**3/u.sr).value

# Simulate 1000 populations, each containing 1000 events
mock_m1s = np.zeros((10,1000)) # (number of pops, number of events per pop)
mock_m2s = np.zeros((10,1000))
mock_zs = np.zeros((10,1000))
mock_s1zs = np.zeros((10,1000))
mock_s2zs = np.zeros((10,1000))

# Choose 1000 random draws from the hyperposterior
inds = np.random.choice(np.arange(0,lams.size),size=1000,replace=False)
nEff = np.zeros(1000)

# Loop across draws
for i,ind in enumerate(inds):
    
    if i%100==0:
        print(i)

    # Read off the hyperposterior sample
    lam = lams[ind]
    mmin = mmins[ind]
    mmax = mmaxs[ind]
    alpha_m = mass_alphas[ind]
    beta_m = mass_betas[ind]
    sig_m = sig_ms[ind]
    mu_m = mu_ms[ind]
    delta_m = delta_ms[ind]
    a = spin_alphas[ind]
    b = spin_betas[ind]
    sigt = sig_ts[ind]
    zeta = zetas[ind]
    kappa = kappas[ind]
    
    #######
    # First, we need to build the probability distribution on m1
    # We'll separately compute the power-law and gaussian pices and combine,
    # then apply low-mass smoothing, then re-normalize
    #######

    # Define a reference grid over primary masses
    ref_ms = np.linspace(mmin,mmax,400)
    
    # Power law m1 component, normalized on (mmin,mmax)
    p_m1_powerLaw = (1.-alpha_m)*np.power(ref_ms,-alpha_m)/(np.power(mmax,1.-alpha_m)-np.power(mmin,1.-alpha_m))
    
    # Gaussian m1 component, normalized over -inf to inf
    p_m1_gaussian = (1./np.sqrt(2.*np.pi*sig_m**2.))*np.exp(-0.5*(ref_ms-mu_m)*(ref_ms-mu_m)/sig_m**2.)

    # Apply smooth turn-on at masses below mmin + delta_m
    smoothingFactors = np.ones(ref_ms.size)
    masses_to_smooth = ref_ms[(ref_ms<mmin+delta_m)]
    smoothingFactors[(ref_ms<mmin+delta_m)] = 1./(np.exp(delta_m/(masses_to_smooth-mmin) - delta_m/(delta_m-(masses_to_smooth-mmin)))+1.)

    # Incorporate smoothing and re-normalize
    pop_m1 = ((1.-lam)*p_m1_powerLaw + lam*p_m1_gaussian)*smoothingFactors
    pop_m1 /= np.trapz(pop_m1,ref_ms)
    
    p_m1_new = np.interp(m1_det,ref_ms,pop_m1)
    p_m1_new[m1_det<=mmin] = 0
    p_m1_new[m1_det>=mmax] = 0

    # Get the new population priors on m2
    p_m2_new = (1.+beta_m)*np.power(m2_det,beta_m)/(np.power(m1_det,1.+beta_m)-np.power(mmin,1.+beta_m))
    p_m2_new[m2_det>mmax] = 0

    # A handful of samples have m1_det sufficiently close to mmin that the denominator in p_m2_new goes to zero
    # In this case, neglect these samples
    p_m2_new[np.isinf(p_m2_new)] = 0
    
    p_z_new = dVdz_det*np.power(1+z_det,kappa-1.)
    
    # Form probabilities on spin tilts and magnitudes
    p_a1_new = np.power(a1_det,a-1)*np.power(1.-a1_det,b-1)
    p_a2_new = np.power(a2_det,a-1)*np.power(1.-a2_det,b-1)

    cost1_gaussian = np.exp(-0.5*np.power(cost1_det-1.,2.)/sigt**2.)/np.sqrt(2.*np.pi*sigt**2.)/\
            (0.5*erf((1.-1.)/np.sqrt(2.*sigt**2.))-0.5*erf((-1.-1.)/np.sqrt(2.*sigt**2.)))
    cost2_gaussian = np.exp(-0.5*np.power(cost2_det-1.,2.)/sigt**2.)/np.sqrt(2.*np.pi*sigt**2.)/\
            (0.5*erf((1.-1.)/np.sqrt(2.*sigt**2.))-0.5*erf((-1.-1.)/np.sqrt(2.*sigt**2.)))
    p_cost1_new = zeta*cost1_gaussian + (1.-zeta)*0.5
    p_cost2_new = zeta*cost2_gaussian + (1.-zeta)*0.5
    
    # Define new weights as the ratio between target prior and old prior
    ps_draw = p_m1_new*p_m2_new*p_z_new*p_a1_new*p_a2_new*p_cost1_new*p_cost2_new*pop_reweight
    ps_draw[ps_draw<0] = 0
    ps_draw /= np.sum(ps_draw)

    nEff[i] = np.sum(ps_draw)**2/np.sum(ps_draw**2)
                    
    try:
        observed_ind = np.random.choice(np.arange(m1_det.size),size=10,p=ps_draw,replace=False)
    except ValueError:
        print(ind)
        print(ps_draw)
        sys.exit(1)
    mock_m1s[:,i] = m1_det[observed_ind]
    mock_m2s[:,i] = m2_det[observed_ind]
    mock_zs[:,i] = z_det[observed_ind]
    mock_s1zs[:,i] = s1z_det[observed_ind]
    mock_s2zs[:,i] = s2z_det[observed_ind]


# Set up figure
fig = plt.figure(figsize=(14,4))

######################
# Subplot 1: m1 vs. q
######################

ax = fig.add_subplot(131)
log_m1s = np.log10(mock_m1s).reshape(-1)
qs = (mock_m2s/mock_m1s).reshape(-1)
zs =  mock_zs.reshape(-1)

xmin=0.9
xmax=2.5
ymin=0.
ymax=1

m_grid = np.linspace(xmin,xmax,120)
q_grid = np.linspace(ymin,ymax,119)
dm = m_grid[1] - m_grid[0]
dq = q_grid[1] - q_grid[0]

# Build KDE and evaluate at grid points
# Note that we want reflection symmetry at q=1
event_kde = gaussian_kde([log_m1s,qs])
M,Q = np.meshgrid(m_grid,q_grid)
heights = event_kde([M.reshape(-1),Q.reshape(-1)]) + event_kde([M.reshape(-1),2.-Q.reshape(-1)])

# Reshape and normalize KDE estimates
# Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
heights = heights.reshape(M.shape)
heights /= np.sum(heights)*dm*dq
heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
cdf = np.cumsum(heights_large_to_small)*dm*dq

h90 = np.interp(0.9,cdf,heights_large_to_small)
h80 = np.interp(0.8,cdf,heights_large_to_small)
h70 = np.interp(0.7,cdf,heights_large_to_small)
h60 = np.interp(0.6,cdf,heights_large_to_small)
h50 = np.interp(0.5,cdf,heights_large_to_small)
h40 = np.interp(0.4,cdf,heights_large_to_small)
h30 = np.interp(0.3,cdf,heights_large_to_small)
h20 = np.interp(0.2,cdf,heights_large_to_small)
h10 = np.interp(0.1,cdf,heights_large_to_small)

ax.contour(m_grid,q_grid,heights,levels=(h90,h70,h50,h30,h10,np.inf),cmap=None,colors='#2e2e2e',linewidths=1)

ax.set_xlabel(r'$m_1\,[M_\odot]$',fontsize=18)
ax.set_ylabel(r'$q$',fontsize=18)
ax.xaxis.grid(True,which='major',ls=':')
ax.yaxis.grid(True,which='major',ls=':')
ax.tick_params(labelsize=14)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

ax.xaxis.set_ticks(np.log10([10,30,100]))
ax.xaxis.set_ticklabels([10,30,100])

for cat, sampleDict in sampleDicts.items():
    for key in sampleDict:
        # Read samples
        log_m1s = np.log10(sampleDict[key]['m1'])
        m2s = sampleDict[key]['m2']
        qs = sampleDict[key]['m2']/sampleDict[key]['m1']
        
        # Build KDE and evaluate at grid points
        # Note that we want reflection symmetry at q=1
        event_kde = gaussian_kde([log_m1s,qs])
        M,Q = np.meshgrid(m_grid,q_grid)
        heights = event_kde([M.reshape(-1),Q.reshape(-1)]) + event_kde([M.reshape(-1),2.-Q.reshape(-1)])

        # Reshape and normalize KDE estimates
        # Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
        heights = heights.reshape(M.shape)
        heights /= np.sum(heights)*dm*dq
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dm*dq
        h90 = np.interp(0.9,cdf,heights_large_to_small)        

        # Plot!
        ax.contourf(m_grid,q_grid,heights,levels=(h90,np.inf),colors=colors[cat],alpha=0.3,rasterize=True)
        ax.contour(m_grid,q_grid,heights,levels=(h90,np.inf),colors='black',linewidths=0.2,alpha=0.3,rasterize=True)

#########################
# Subplot 2: chieff vs. Mc
#########################


ax = fig.add_subplot(132)
xs = ((mock_m1s*mock_s1zs + mock_m2s*mock_s2zs)/(mock_m1s+mock_m2s)).reshape(-1)
etas = mock_m1s*mock_m2s/(mock_m1s+mock_m2s)**2
log_Mcs = np.log10((etas**(3./5.)*(mock_m1s+mock_m2s))).reshape(-1)

xmin=-1.
xmax=1.
ymin=0.5
ymax=2.

x_grid = np.linspace(xmin,xmax,120)
m_grid = np.linspace(ymin,ymax,119)
dx = x_grid[1] - x_grid[0]
dq = m_grid[1] - m_grid[0]

# Build KDE and evaluate at grid points
event_kde = gaussian_kde([xs,log_Mcs])
X,M = np.meshgrid(x_grid,m_grid)
heights = event_kde([X.reshape(-1),M.reshape(-1)])

# Reshape and normalize KDE estimates
# Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
heights = heights.reshape(M.shape)
heights /= np.sum(heights)*dx*dm
heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
cdf = np.cumsum(heights_large_to_small)*dx*dm

h90 = np.interp(0.9,cdf,heights_large_to_small)
h80 = np.interp(0.8,cdf,heights_large_to_small)
h70 = np.interp(0.7,cdf,heights_large_to_small)
h60 = np.interp(0.6,cdf,heights_large_to_small)
h50 = np.interp(0.5,cdf,heights_large_to_small)
h40 = np.interp(0.4,cdf,heights_large_to_small)
h30 = np.interp(0.3,cdf,heights_large_to_small)
h20 = np.interp(0.2,cdf,heights_large_to_small)
h10 = np.interp(0.1,cdf,heights_large_to_small)

ax.contour(x_grid,m_grid,heights,levels=(h90,h70,h50,h30,h10,np.inf),vmin=-0.1,vmax=h10,colors='#2e2e2e',linewidths=1.)

ax.set_xlabel(r'$\chi_\mathrm{eff}$',fontsize=18)
ax.set_ylabel(r'$\mathcal{M}\,[M_\odot]$',fontsize=18)
ax.xaxis.grid(True,which='major',ls=':')
ax.yaxis.grid(True,which='major',ls=':')
ax.tick_params(labelsize=14)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

ax.yaxis.set_ticks(np.log10([3,10,30,100]))
ax.yaxis.set_ticklabels([3,10,30,100])

for cat, sampleDict in sampleDicts.items():
    for key in sampleDict:
        
        # Read samples
        log_mcs = np.log10(sampleDict[key]['Mc'])
        chis = sampleDict[key]['Xeff']
        
        # Build KDE and evaluate at grid points
        event_kde = gaussian_kde([chis,log_mcs])
        X,M = np.meshgrid(x_grid,m_grid)
        heights = event_kde([X.reshape(-1),M.reshape(-1)])

        # Reshape and normalize KDE estimates
        # Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
        heights = heights.reshape(M.shape)
        heights /= np.sum(heights)*dx*dm
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dx*dm
        h90 = np.interp(0.9,cdf,heights_large_to_small)        

        # Plot!
        ax.contourf(x_grid,m_grid,heights,levels=(h90,np.inf),colors=colors[cat],alpha=0.3,rasterized=True)
        ax.contour(x_grid,m_grid,heights,levels=(h90,np.inf),colors='black',linewidths=0.2,alpha=0.3,rasterized=True)

#########################
# Subplot 3: m1 vs. z
#########################

ax = fig.add_subplot(133)
log_m1s = np.log10(mock_m1s).reshape(-1)
qs = (mock_m2s/mock_m1s).reshape(-1)
zs =  mock_zs.reshape(-1)

xmin=0
xmax=2.0
ymin=0.9
ymax=2.5

z_grid = np.linspace(xmin,xmax,120)
M_grid = np.linspace(ymin,ymax,119)
dz = z_grid[1] - z_grid[0]
dM = M_grid[1] - M_grid[0]

# Build KDE and evaluate at grid points
# We want reflection symmetry across m=0
event_kde = gaussian_kde([zs,log_m1s])
Z,M = np.meshgrid(z_grid,M_grid)
heights = event_kde([Z.reshape(-1),M.reshape(-1)]) + event_kde([Z.reshape(-1),-M.reshape(-1)])

# Reshape and normalize KDE estimates
# Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
heights = heights.reshape(M.shape)
heights /= np.sum(heights)*dz*dM
heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
cdf = np.cumsum(heights_large_to_small)*dz*dM

h90 = np.interp(0.9,cdf,heights_large_to_small)
h80 = np.interp(0.8,cdf,heights_large_to_small)
h70 = np.interp(0.7,cdf,heights_large_to_small)
h60 = np.interp(0.6,cdf,heights_large_to_small)
h50 = np.interp(0.5,cdf,heights_large_to_small)
h40 = np.interp(0.4,cdf,heights_large_to_small)
h30 = np.interp(0.3,cdf,heights_large_to_small)
h20 = np.interp(0.2,cdf,heights_large_to_small)
h10 = np.interp(0.1,cdf,heights_large_to_small)

ax.contour(z_grid,M_grid,heights,levels=(h90,h70,h50,h30,h10,np.inf),cmap=None,vmin=-0.1,vmax=h10,colors='#2e2e2e',linewidths=1.,label='GWTC-3 BBH Predictions')

ax.set_xlabel(r'$z$',fontsize=18)
ax.set_ylabel(r'$m_1\,[M_\odot]$',fontsize=18)
ax.xaxis.grid(True,which='major',ls=':')
ax.yaxis.grid(True,which='major',ls=':')
ax.tick_params(labelsize=14)
ax.set_xlim(xmin,xmax)
ax.set_ylim(ymin,ymax)

ax.yaxis.set_ticks(np.log10([10,30,100]))
ax.yaxis.set_ticklabels([10,30,100])

for cat, sampleDict in sampleDicts.items():
    for key in sampleDict:  
    # Read samples
        m1s = sampleDict[key]['m1']
        zs = sampleDict[key]['z']
        log_m1s = np.log10(m1s)

        # Build KDE and evaluate at grid points
        event_kde = gaussian_kde([zs,log_m1s])
        Z,M = np.meshgrid(z_grid,M_grid)
        heights = event_kde([Z.reshape(-1),M.reshape(-1)])

        # Reshape and normalize KDE estimates
        # Then compute a 1D CDF and find the probability density bounding 90% of our posterior probability
        heights = heights.reshape(M.shape)
        heights /= np.sum(heights)*dz*dM
        heights_large_to_small = np.sort(heights.reshape(-1))[::-1]
        cdf = np.cumsum(heights_large_to_small)*dz*dM
        h90 = np.interp(0.9,cdf,heights_large_to_small)        

        # Plot!
        ax.contourf(z_grid,M_grid,heights,levels=(h90,np.inf),colors=colors[cat],alpha=0.3,rasterized=True)
        ax.contour(z_grid,M_grid,heights,levels=(h90,np.inf),colors='black',linewidths=0.2,alpha=0.3,rasterized=True)

"""
ax.legend([Rectangle((0,0),1,1,color=colors['IAS']),
           Rectangle((0,0),1,1,color=colors['OGC']),
           Rectangle((0,0),1,1,color=colors['LVK']),
           #Rectangle((0,0),1,1,facecolor='white',edgecolor='black')
          ],
          ["IAS", "OGC", "GWTC-3 FAR $> 1$/year",#"GWTC-3 Predictions"
          ],
        loc='lower right',frameon=False,fontsize=14)
"""

ax.legend([Rectangle((0,0),1,1,color=colors['IAS']),
           Rectangle((0,0),1,1,color=colors['OGC'])
          ],
          ["IAS", "OGC"
          ],
        loc='lower right',frameon=False,fontsize=14)

plt.tight_layout()
plt.savefig('../../../figures/O4a_vs_other_blob_comparison.pdf',bbox_inches='tight',dpi=200)
