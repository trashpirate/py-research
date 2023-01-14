import matplotlib.cbook as mplbook

import matplotlib as mpl

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.cm as cm
import matplotlib.colors as colors

from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D

# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# from mpl_toolkits.axes_grid1.colorbar import colorbar

from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, PowerNorm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.patches import Patch, PathPatch

import seaborn as sns
from cycler import cycler

import numpy as np

import warnings

warnings.filterwarnings("ignore", category=mplbook.mplDeprecation)

minsize = 1.1811  # minimal size
singlecol = 3.46  # single column
oneandhalf = 4.72  # 1.5 column
doublecol = 7.086  # double column

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
    }
)
# plt.rcParams['font.sans-serif'] = 'Helvetica'
# plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "DejaVu Sans"
plt.rcParams["mathtext.it"] = "DejaVu Sans"
plt.rcParams["mathtext.sf"] = "DejaVu Sans:bold"
plt.rcParams["mathtext.tt"] = "DejaVu Sans"
plt.rcParams["mathtext.cal"] = "DejaVu Sans:italic"

plt.rcParams["image.cmap"] = "afmhot"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.grid"] = False
plt.rcParams["grid.color"] = "k"
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.alpha"] = 0.5

# matplotlib.rcParams.update({'font.size': 22, 'font.weight': 'bold'})
plt.rcParams["figure.figsize"] = [4, 4]
# plt.rcParams['figure.autolayout']=True
# plt.rcParams['figure.constrained_layout.h_pad'] = 0
# plt.rcParams['figure.constrained_layout.w_pad'] = 0
# plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300

plt.rcParams["font.size"] = 7
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["legend.frameon"] = False
plt.rcParams["figure.titlesize"] = 7
plt.rcParams["axes.titlesize"] = 7
plt.rcParams["axes.labelsize"] = 7
plt.rcParams["xtick.labelsize"] = 6
plt.rcParams["ytick.labelsize"] = 6

# plt.rcParams['axes.spines.top'] = False
# plt.rcParams['axes.spines.right'] = False
# plt.rcParams['axes.prop_cycle'] = cycler(color=sns.dark_palette("blue",n_colors=10))

plt.rcParams["image.cmap"] = "viridis"

plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
# plt.rcParams['text.usetex'] = True


defaultpalette = cm.viridis

R = "darkred"
B = "darkblue"
G = "darkgreen"
Y = "gold"
O = "darkorange"

blues = sns.light_palette("darkblue", n_colors=5, reverse=True)
cyans = sns.light_palette("darkcyan", n_colors=5, reverse=True)
greens = sns.light_palette("darkgreen", n_colors=5, reverse=True)
golds = sns.light_palette("orange", n_colors=5, reverse=True)
oranges = sns.light_palette("orangered", n_colors=5, reverse=True)
reds = sns.light_palette("darkred", n_colors=5, reverse=True)
purples = sns.light_palette("indigo", n_colors=5, reverse=True)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def set_fontsize(fig=8, title=8, legend=6):
    plt.rcParams["font.size"] = fig - 2
    plt.rcParams["legend.fontsize"] = legend
    plt.rcParams["figure.titlesize"] = title
    plt.rcParams["axes.titlesize"] = fig
    plt.rcParams["axes.labelsize"] = fig - 1
    plt.rcParams["xtick.labelsize"] = fig - 2
    plt.rcParams["ytick.labelsize"] = fig - 2


def set_figsize(w, h):
    plt.rcParams["figure.figsize"] = [w, h]


def set_colorcycle(palette, n=5):
    plt.rcParams["axes.prop_cycle"] = cycler(color=palette)


def style_boxplot(ax, labels=None, loc=1):
    colors = []
    box_patches = [patch for patch in ax.patches if type(patch) == mpl.patches.PathPatch]

    for i, artist in enumerate(box_patches):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        r, g, b, a = artist.get_facecolor()
        col = (r, g, b, a)
        artist.set_edgecolor(col)
        artist.set_facecolor((r, g, b, 0.7))

        for j in range(i * 6, i * 6 + 6):
            line = ax.lines[j]
            line.set_color(col)
            line.set_mfc(col)
            line.set_mec(col)

        colors.append([col, (r, g, b, 0.7)])

    handles = []
    if labels is not None:
        for c, label in enumerate(labels):
            handles.append(Patch(facecolor=colors[c][1], edgecolor=colors[c][0], label=label))
    ax.legend(handles=handles, loc=loc)


def autolabel(rects, ax, offset=3):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            "%d" % round(height),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, offset),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )


def adjust_box_widths(g, fac):
    """
    Adjust the withs of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5 * (xmin + xmax)
                xhalf = 0.5 * (xmax - xmin)

                # setting new width of box
                xmin_new = xmid - fac * xhalf
                xmax_new = xmid + fac * xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def change_width(ax, new_value):
    for patch in ax.patches:
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * 0.5)


def add_colorbar(im, aspect=20, pad_fraction=0.1, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("top", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def grouped_barplot(X, Y, E=None, colors=None, labels=None, ax=None):
    import numpy as np

    N = len(X)
    if colors == None:
        colors = sns.color_palette("husl", N)
    if E == None:
        E = np.zeros(N)
    if labels == None:
        labels = [""] * N

    ind = np.arange(Y[0].shape[-1])  # the x locations for the groups
    width = 0.8 / N  # the width of the bars

    if ax == None:
        ax = plt.gca()
    for i in range(N):
        rect = ax.bar(ind + width * i - 0.4, Y[i], width, yerr=E[i], align="edge", color=colors[i], label=labels[i])
    print(ind)
    ax.set_xticks(ind)
    ax.set_xticklabels(X[0])


def set_layout(fig, l=0.55, h=0.45):
    size = fig.get_size_inches()
    axes = fig.get_axes()
    xx = l / size[0]
    yy = h / size[1]
    for ax in axes:
        pos = ax.get_position()  # get the original position
        pos_new = [xx, yy, pos.width * 0.95, pos.height * 0.95]
        ax.set_position(pos_new, which="both")
        pos = ax.get_position()
        xx = pos.x0 + pos.width + 0.1 / size[0]
        yy = pos.y0
