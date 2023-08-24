'''SFH house - case study for VIC & NRW - Post-processing'''
# AUTHOR: Mauricio Carcamo M.

#### file - database handling and housekeeping
import os
import scienceplots
from pathlib import Path
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import matplotlib as mpl
import numpy as np
# print(plt.style.available)
plt.style.use(['science'])
# plt.rcParams.update({
# "text.usetex": False,
# "font.family": "sans-serif",
# "font.sans-serif": ["Helvetica"]})

# Set the default color cycle
colors = ['purple','red','green','blue','orange','cyan','olive','magenta','grey','sandybrown','royalblue'] 
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 



def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, 
                     colors=None, grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will 
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        color = colors[i] if colors is not None else None
        axes.append(plt.bar(ind, row_data, bottom=cum_size, 
                            label=series_labels[i], color=color))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w/2, bar.get_y() + h/2, 
                         value_format.format(h), ha="center", 
                         va="center")

#### results import

parent_path = Path(__file__).parents[0]
results_path = parent_path / '_results'

os.chdir(str(parent_path))

###### list of files
#Decen
dirfile_VIC = [f for f in results_path.glob("*VIC_SFH_detailed.pk")]
dirfile_NRW = [f for f in results_path.glob("*NRW_SFH_detailed.pk")]


###### Database
with open(f'{dirfile_NRW[0]}', 'rb') as f: # last generated pickle
    db_NRW = pickle.load(f)
with open(f'{dirfile_VIC[0]}', 'rb') as f: # last generated pickle
    db_VIC = pickle.load(f)

# print(db_NRW)

##### FIG 1
##DB Filtering

elec_ind_NRW = 18.33
elec_cons_NRW = 32.79
elec_fac_NRW = elec_ind_NRW/elec_cons_NRW
gas_cons_NRW = 8.06
gas_ind_NRW = 5.03
gas_fac_NRW = gas_ind_NRW/gas_cons_NRW

elec_ind_VIC = 27.09/1.5
elec_cons_VIC = 33.55/1.5
elec_fac_VIC = elec_ind_VIC/elec_cons_VIC
gas_cons_VIC = 10.24/1.5
gas_ind_VIC = 7.70/1.5
gas_fac_VIC = gas_ind_VIC/gas_cons_VIC


print(db_NRW)
db_NRW_base = db_NRW.fillna(0)
db_VIC_base = db_VIC.fillna(0)
# print(list(db_NRW_base.columns))
# print(list(db_NRW_base['LCOE']))
# print(list(db_NRW_base['tac']))
print(db_VIC)
print(list(db_VIC_base['LCOE']))
print(list(db_VIC_base['tac']))
# print(list(db_NRW_base['el_price']))