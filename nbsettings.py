## Matplotlib settings for Jupyter displays.

import matplotlib.pyplot as plt

plt.rcParams['savefig.dpi'] = 75
plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 8, 4
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

# from IPython.display import HTML
# HTML('''<style>
# .output_png {
#     display: table-cell;
#     text-align: center;
#     vertical-align: middle;
# }
# </style>
# <script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# Python source code for figures is hidden by default.
# To toggle code display on/off, click <a href="javascript:code_toggle()">here</a>.''')
