import numpy as np 
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# filepaths for saving outputs
figs_folder = '../figures/'
DATADIR = '../data_release/'

# where the GWTC-3 official results are stored
gwtc3_folder = '../gwtc3_data/analyses/'

# default full-page figure width
FIG_WIDTH = 7

# define commonly used colors in my plots; 
# confirmed that utilized combinations are colorblind friendly
# using https://davidmathlogic.com/colorblind
color_gwtc3 = '#000000'    # black
color_default = '#2798e3'  # blue
color_bspline = '#A1C935'  # light green
color_spinA = '#145A8D'    # darker blue
color_spinB = '#84C5EA'    # lighter blue
color_skewnorm_eff = '#FF5733'   # orange/red
color_gaussian_eff = '#6a6bad'   # purple
color_gaussian_eff_2 = '#57B578' # green
color_bgp = '#2D9CDB'  # bright blue 
color_corr = '#582500' # maroon


def setup(): 

    from matplotlib import rc, rcParams

    rc_params = {
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
         "savefig.dpi": 300,
    }
    rcParams.update(rc_params)
    rc('axes', linewidth=0.5)
    rc('text', usetex=True)
    rc('font', family='serif')
    rcParams["xtick.major.pad"] = "6"
    rcParams["ytick.major.pad"] = "6"

    import seaborn as sns
    sns.set_palette("colorblind")


def get_rgb_from_hex(hex_color): 

    if hex_color.startswith("#"):
        hex_color = hex_color[1:]

    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def get_hex_from_rgb(rgb_color): 
    return f"#{''.join(f'{c:02x}' for c in rgb_color)}"

def darken_color(hex_color, factor=0.5):
    """
    Darkens a given hex color by a specified factor.

    Args:
        hex_color (str): The hex color code (e.g., "#FF0000").
        factor (float): The darkening factor (0.0 - 1.0, default 0.5).

    Returns:
        str: The darkened hex color code.
    """
    rgb = get_rgb_from_hex(hex_color)
    darkened_rgb = tuple(int(c * factor) for c in rgb)
    return get_hex_from_rgb(darkened_rgb)


def lighten_color(hex_color, factor=0.2):
    """
    Lightens the given hex color by blending it with white.
    
    :param hex_color: The original color in hex format, e.g., '#3498db'.
    :param factor: A float between 0 and 1 indicating how much to lighten the color.
                   0 means no change, 1 means fully white.
    :return: The lightened color as a hex string.
    """
    
    # Convert hex to RGB
    r, g, b = get_rgb_from_hex(hex_color)
    
    # Lighten each channel by blending with white
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    
    # Convert back to hex
    return f"#{r:02x}{g:02x}{b:02x}"


def make_colormap(hex_color): 

    # Convert torgb
    color = get_rgb_from_hex(hex_color)
    color = tuple([c/256 for c in color])

    # Lightened for min 
    lightened_color = lighten_color(hex_color, factor=0.93)
    lightened_color = get_rgb_from_hex(lightened_color)
    lightened_color = tuple([c/256 for c in lightened_color])

    # Darkened for max 
    darkened_color = darken_color(hex_color, factor=0.6)
    darkened_color = get_rgb_from_hex(darkened_color)
    darkened_color = tuple([c/256 for c in darkened_color])

    # Generate a monochromatic colormap based on the custom color
    colors = [lightened_color, color, darkened_color] 

    # Create the colormap using LinearSegmentedColormap
    custom_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

    return custom_cmap


def format_inputs(pos, traces):
    
    # make sure they are numpy arrays and not lists
    if isinstance(pos, list): 
        pos = np.asarray(pos)
    if isinstance(traces, list): 
        traces = np.asarray(traces)

    # format inputs to be right shapes
    pos_plot = pos.flatten() if len(pos.shape) > 1 else pos
    if len(pos_plot) == traces.shape[0]: 
        traces = np.transpose(traces)
    return pos_plot, traces


def plot_traces(ax, pos, traces, n_traces=1000, color='#41a2e4', just_quants=False, 
                fill_between=False, normed=True, ls=None):

    # format inputs to be right sizes
    pos_plot, traces = format_inputs(pos, traces)

    # remove nans
    traces = traces[~np.isnan(traces).any(axis=-1)]

    # numerially normalize the traces if they were not normalized in the 
    # given popsummary file
    if not normed: 
        dx = pos_plot[1] - pos_plot[0]
        traces_ = [y/(np.sum(y) * dx) for y in traces]
        traces = traces_

    # calculate quantiles
    quants = np.quantile(traces, (0.05, 0.5, 0.95), axis=0)    

    # option to plot all the traces / fill in
    if not just_quants: 

        if fill_between:
            ax.fill_between(pos_plot, quants[0], quants[2], color=color, alpha=0.5, zorder=1)
        else:
            idxs = np.random.choice(len(traces), size=n_traces)
            ax.plot(pos_plot, traces[idxs].T, alpha=0.03, color=color, lw=0.5, zorder=1)

        # for quantiles
        color2 = darken_color(color)
        if ls is None:
            ls = '-'
        lw = 1

    # option to plot only the quantiles
    else: 
        # for quantiles
        color2 = color
        if ls is None:
            ls = '--'
        lw = 1.5

        ax.plot(pos_plot, quants[0], color=color2, lw=0.75, ls=ls)
        ax.plot(pos_plot, quants[2], color=color2, lw=0.75, ls=ls)


    # plot the median
    ax.plot(pos_plot, quants[1], color=color2, lw=lw, ls=ls)

    # Set the rasterization zorder threshold
    ax.set_rasterization_zorder(1.5)


def get_chieff_mins(pos, traces, n_points=5000): 

    # format inputs to be right sizes
    pos, traces = format_inputs(pos, traces)

    idxs = np.random.choice(len(traces), size=n_points)
    traces_downsampled = traces[idxs]

    min_chieffs = np.zeros(n_points)

    for i, trace in enumerate(traces_downsampled): 
        min_chieffs[i] = get_percentile(pos, trace, 0.01)

    return min_chieffs


def get_percentile(x, y, percentile): 

    ntarget = 1000

    # x must be sorted 
    random_x = np.random.uniform(x[0], x[-1], size=ntarget*100)

    # linearly interpolate to get the random y's
    random_y = np.interp(random_x, x, y, left=0, right=0)

    # perform weighted draws 
    x_choice = np.random.choice(random_x, p=random_y/np.sum(random_y), size=ntarget)

    # calculate the value of the weighted x draws at the given percentile
    p = np.quantile(x_choice, percentile)

    return p


def get_fraction_negative(x,y_array): 

    x, y_array = format_inputs(x,y_array)

    f_array = np.zeros(len(y_array))
    for i,y in enumerate(y_array):
        f_array[i] = np.sum(y[x<0]) /  np.sum(y)
    return f_array

def print_quantiles(p): 
    q_05, q_50, q_95 = np.quantile(p, [0.05, 0.5, 0.95])
    upper_error = q_95 - q_50
    lower_error = q_50 - q_05
    print(f'{q_50:.2f}^{{+{upper_error:.2f}}}_{{-{lower_error:.2f}}}')


def compute_contour_levels(ppd, levels=[0.5]):
    sorted_ppd = np.sort(ppd.flatten())[::-1]
    cumsum = np.cumsum(sorted_ppd)
    cumsum /= cumsum[-1]  # Normalize to 1
    return [sorted_ppd[np.searchsorted(cumsum, level)] for level in levels]

def cornerplot(axes, data, parameters, run_label, limits_dict, hist_kws, nbins=30): 

    ## 1D HISTS
    i=0; k = parameters[0]; 
    axes[i][i].hist(data[k], **hist_kws, bins=np.linspace(*limits_dict[k], nbins), label=run_label)
    for ax in axes[:,i]: 
        ax.set_xlim(*limits_dict[k])
    
    for _i, k in enumerate(parameters[1:]): 
        i = _i + 1
        axes[i][i].hist(data[k], **hist_kws, bins=np.linspace(*limits_dict[k], nbins))
        for ax in axes[:,i]: 
            ax.set_xlim(*limits_dict[k])
    
    ## 2D HISTS
    for i, ki in enumerate(parameters): 
        for j, kj in enumerate(parameters): 
            if i > j: 
                sns.kdeplot(data=data, y=ki, x=kj, ax=axes[i][j], color=hist_kws['color'], levels=[0.5,0.9])

def cornerplot_axes_setup(axes, param_labels): 

    nparams = len(axes[0])

    # axes labeling
    for i in range(nparams): 
        for j in range(nparams): 
            if i < j: 
                axes[i][j].set_visible(False)
                
            if i!=nparams-1: 
                axes[i][j].set_xticklabels('')
                axes[i][j].set_xlabel('')
    
            if j!=0: 
                axes[i][j].set_yticklabels('')
                axes[i][j].set_ylabel('')
                
            axes[i][j].grid(color='silver', alpha=0.5, ls=':')      
            
    for ax,l in zip(axes[nparams-1],param_labels): 
        ax.set_xlabel(l)
    for ax,l in zip(axes[:,0],param_labels): 
        ax.set_ylabel(l)
    axes[0][0].set_yticklabels('')
    axes[0][0].set_ylabel('')
    
def chi_p_max(q, chi_eff, a_max=1.0):
    """
    Compute chi_p,max as defined by the piecewise function.

    Parameters:
    - q (float): Mass ratio (m2/m1 <= 1, so 0 < q <= 1)
    - chi_eff (float): Effective spin parameter
    - a_max (float): Maximum dimensionless spin (default is 1.0)

    Returns:
    - float: chi_p,max
    """
    chi_eff_abs = abs(chi_eff)
    threshold = q / (1 + q)

    if chi_eff_abs / a_max <= threshold:
        return a_max
    else:
        inner_term = ((1 + q) * chi_eff_abs - q * a_max) ** 2
        return np.sqrt(a_max**2 - inner_term)