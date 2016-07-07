## Matplotlib settings for Jupyter displays.

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 10, 5
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['ytick.major.width'] = 1.4
plt.rcParams['ytick.minor.width'] = 1.4
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['xtick.major.width'] = 1.4
plt.rcParams['xtick.minor.width'] = 1.4
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 8
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
plt.rcParams['text.latex.preamble'] = "\\usepackage{subdepth}, \\usepackage{type1cm}"

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'png')
