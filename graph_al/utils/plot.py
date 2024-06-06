from numpy.typing import NDArray
import matplotlib.pyplot as plt

import numpy as np

def figure_to_numpy(fig) -> NDArray:
    """Converst a figure to a numpy array.

    Parameters
    ----------
    fig : plt.figure.Figure
        The figure to convert.

    Returns
    -------
    NDArray
        The resulting numpy array as rgba values.
    """
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot

def setup_matplotlib(updates={}):
    """Sets up matplotlib for plotting in nice pgfs for a paper
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        'grid.linewidth': 0.5,
        'grid.color': '.8',
        'axes.linewidth': 0.75,
        'axes.edgecolor': '.7',
        'lines.linewidth': 1.0,
        'xtick.major.width': 0.75,
        'ytick.major.width': 0.75,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.minor.width': 0.75,
        'ytick.minor.width': 0.75,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
        'lines.linewidth': 1.0,
        'legend.fancybox': False,
    } | {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,  # LaTeX default is 10pt font.
        "font.size": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.latex.preamble": r"""
            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}
            \usepackage{amsmath}
            \newcommand*{\mat}[1]{\boldsymbol{#1}}
            """,
        "pgf.preamble": r"""
            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}
            \usepackage{amsmath}
            \newcommand*{\mat}[1]{\boldsymbol{#1}}
            """,
    } | updates)