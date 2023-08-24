'''SFH house - case study for VIC & NRW - Post-processing'''
# AUTHOR: Mauricio Carcamo M.

#### file - database handling and housekeeping
from palettable import tableau
import scienceplots
import os
from pathlib import Path
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
import matplotlib as mpl
import numpy as np

#####
## Colors and plot style
#####

plt.style.use(['science'])
# plt.style.use(['nature'])

### Save for this for final version
# plt.style.use(['nature'])
# mpl.rc('font', **{'family':'sans-serif', 'sans-serif':['Tex Gyre Heros']})
# mpl.rc('text', usetex=False)

# mpl.rc('text.latex', preamble=
#        r'\usepackage[utf8]{inputenc}'
#        r'\usepackage{siunitx}'   # i need upright \micro symbols, but you need...
#        r'\sisetup{detect-all}'   # ...this to force siunitx to actually use your fonts
#     #    r'\usepackage{helvet}'   # set the normal font here
#        r'\usepackage{tgheros}'
#        r'\renewcommand{\familydefault}{\sfdefault}'
#        r'\usepackage[eulergreek]{sansmath}'  # load up the sansmath so that math -> helvet
#        r'\sansmath')               # <- tricky! -- gotta actually tell tex to use!


linestyle = ['-', '--', ':', '-.',(0, (3, 5, 1, 5, 1, 5))]
colors = ['#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd', #12 x 3
'#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd',
'#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd','#0d49fb', '#e6091c', '#26eb47', '#8936df', '#fec32d', '#25d7fd',
] # SciencePlots colors
markers = ['o','^','s','*','D','P','X']
# colors = None
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) # Using SciencPlots defaultstyle
month_year_formatter = mpl.dates.DateFormatter('%b') #Date format
monthly_locator = mpl.dates.MonthLocator()



#nature requires 5-7 pt
# SMALLER_SIZE = 7
# SMALL_SIZE = 7
# MEDIUM_SIZE = 7
# BIGGER_SIZE = 7

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

SMALLER_SIZE = 7
SMALL_SIZE = 8
MEDIUM_SIZE = 9
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALLER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# axis
## x axis
plt.rc('xtick', **{ 'direction' : 'in',
                    'major.size' : 3,
                    'major.width' : 0.5,
                    'minor.size' : 1.5,
                    'minor.width' : 0.5,
                    'minor.visible' : True,
                    'top' : 'True'})
## y axis
plt.rc('ytick', **{ 'direction' : 'in',
                    'major.size' : 3,
                    'major.width' : 0.5,
                    'minor.size' : 1.5,
                    'minor.width' : 0.5,
                    'minor.visible' : True,
                    'right' : True})

# line widths
plt.rc('axes', **{'linewidth': 0.5})
plt.rc('grid', **{'linewidth': 0.5})
plt.rc('lines', **{'linewidth': 1.0})

#legend
plt.rc('legend', **{'frameon': False})

#savefig
mpl.rc('savefig',**{'bbox': 'tight',
                    'pad_inches':0.05})

# size
mm = 1/25.4 ## mm to inches 
# plt.rcParams['figure.constrained_layout.use'] = True
def figsize(width = 'small',height = 60):
    if type(height) == int:
        if width == 'small':
            return (90*mm,height*mm)
        elif width == 'medium':
            return (140*mm,height*mm)
        elif width == 'large':
            return (180*mm,height*mm)
        elif type(width) == int:
            return (width*mm,height*mm)
        else:
            ValueError('input is width,height. width must be specified as "small", "medium","large" or an int.')
    else:
        ValueError('input is width,height. Height must be an int value')

def draw_text(ax,string, fontsize = MEDIUM_SIZE, loc = 'upper left'):
    """
    Draw two text-boxes, anchored by different corners to the upper-left
    corner of the figure. function for changing referes ax. may be found in FigHolder.py
    """
    from matplotlib.offsetbox import AnchoredText

    if type(string) == str:
        pass
    else:
        ValueError('Text box requires a string as an input.')

    at = AnchoredText(string,
                      loc = loc, prop=dict(size=fontsize), frameon=False,
                      )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

def plot_stacked_bar(data, series_labels, category_labels=None, 
                     show_values=False, value_format="{}", y_label=None, x_label = None, 
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
    
    if x_label:
        plt.xlabel(x_label)

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

def bar_plot(ax, data, group_stretch=0.8, bar_stretch=0.95,
             legend=True, x_labels=True, label_fontsize=8,
             colors=None, barlabel_offset=1,
             bar_labeler=lambda k, i, s: str(round(s, 3))):
    """
    Draws a bar plot with multiple bars per data point.
    :param dict data: The data we want to plot, wher keys are the names of each
      bar group, and items is a list of bar values for the corresponding group.
    :param float group_stretch: 1 means groups occupy the most (largest groups
      touch side to side if they have equal number of bars).
    :param float bar_stretch: If 1, bars within a group will touch side to side.
    :param bool x_labels: If true, x-axis will contain labels with the group
      names given at data, centered at the bar group.
    :param int label_fontsize: Font size for the label on top of each bar.
    :param float barlabel_offset: Distance, in y-values, between the top of the
      bar and its label.
    :param function bar_labeler: If not None, must be a functor with signature
      ``f(group_name, i, scalar)->str``, where each scalar is the entry found at
      data[group_name][i]. When given, returns a label to put on the top of each
      bar. Otherwise no labels on top of bars.
    """
    sorted_data = list(sorted(data.items(), key=lambda elt: elt[0]))
    sorted_k, sorted_v  = zip(*sorted_data)
    max_n_bars = max(len(v) for v in data.values())
    group_centers = np.cumsum([max_n_bars
                               for _ in sorted_data]) - (max_n_bars / 2)
    bar_offset = (1 - bar_stretch) / 2
    bars = defaultdict(list)
    #
    if colors is None:
        colors = {g_name: [f"C{i}" for _ in values]
                  for i, (g_name, values) in enumerate(data.items())}
    #
    for g_i, ((g_name, vals), g_center) in enumerate(zip(sorted_data,
                                                         group_centers)):
        n_bars = len(vals)
        group_beg = g_center - (n_bars / 2) + (bar_stretch / 2)
        for val_i, val in enumerate(vals):
            bar = ax.bar(group_beg + val_i + bar_offset,
                         height=val, width=bar_stretch,
                         color=colors[g_name][val_i])[0]
            bars[g_name].append(bar)
            if  bar_labeler is not None:
                x_pos = bar.get_x() + (bar.get_width() / 2.0)
                y_pos = val + barlabel_offset
                barlbl = bar_labeler(g_name, val_i, val)
                ax.text(x_pos, y_pos, barlbl, ha="center", va="bottom",
                        fontsize=label_fontsize)
    if legend:
        ax.legend([bars[k][0] for k in sorted_k], sorted_k)
    #
    ax.set_xticks(group_centers)
    if x_labels:
        ax.set_xticklabels(sorted_k)
    else:
        ax.set_xticklabels()
    return bars, group_centers

def K2C(temperature):
    """Convert temperature from Kelvin to Celsius."""
    return temperature - 273.15

#########
#### Results import
#########

parent_path = Path(__file__).parents[0]
results_path = parent_path / '_results'


os.chdir(str(parent_path))

# #List of Files and import
file_name_NRW_NoIns = 'NRW_DECEN24d_final_NoIns'
file_name_VIC_NoIns = 'VIC_DECEN24d_final_NoIns'
file_name_NRW_S3 = 'NRW_DECEN24d_final_S3'
file_name_VIC_S3 = 'VIC_DECEN24d_final_S3'

dirfile_VIC_NoIns = [f for f in results_path.glob(f'*{file_name_VIC_NoIns}_post_processed_temp.pk')]
dirfile_NRW_NoIns = [f for f in results_path.glob(f'*{file_name_NRW_NoIns}_post_processed_temp.pk')]
dirfile_VIC_S3 = [f for f in results_path.glob(f'*{file_name_VIC_S3}_post_processed_temp.pk')]
dirfile_NRW_S3 = [f for f in results_path.glob(f'*{file_name_NRW_S3}_post_processed_temp.pk')]


######
### Commodities Prices Data
######

###### commodities costs
elec_cons_NRW_ori = round(32.79/100,4)
elec_ind_NRW_ori = round(18.33/100,4)
elec_fac_NRW = round(elec_ind_NRW_ori/elec_cons_NRW_ori,4)
gas_cons_NRW = round(8.06/100,4)
gas_ind_NRW = round(5.03/100,4)

elec_cons_VIC_ori = round(33.55/1.5/100,4)
elec_ind_VIC_ori = round(27.09/1.5/100,4)
elec_fac_VIC = round(elec_ind_VIC_ori/elec_cons_VIC_ori,4)
gas_cons_VIC = round(10.24/1.5/100,4)
gas_ind_VIC = round(7.70/1.5/100,4)

elec_cons = {'NRW_NoIns': elec_cons_NRW_ori, 'VIC_NoIns': elec_cons_VIC_ori,'NRW_S3': elec_cons_NRW_ori, 'VIC_S3': elec_cons_VIC_ori}
elec_fac = {'NRW': elec_fac_NRW, 'VIC': elec_fac_VIC}
elec_ind = {'NRW': elec_ind_NRW_ori, 'VIC': elec_ind_VIC_ori}
gas_cons = {'NRW': gas_cons_NRW, 'VIC': gas_cons_VIC}
gas_ind = {'NRW': gas_ind_NRW, 'VIC': gas_ind_VIC}

### SFH LCOE & TAC - elec price = [0.1, 0.2, 0.3, 0.4, 0.5, base cost]\
x_SFH = np.arange(-5000, 150000, 1000) # only for plotting/coloring purposes
SFH_NRW_LCOE = [72.87455876557871, 118.17048251691664, 163.4664062682543, 168.3000122546595, 168.4452107587107, 168.19532416435635]
SFH_NRW_TAC = [1127.92725885627, 1829.0019271579479, 2530.0765954596213, 2604.88947999694, 2607.136812309322, 2603.2691538813424]
SFH_VIC_LCOE = [65.03206581590275, 106.78868165739262, 148.5452975719377, 161.3423741098446, 162.58609806912528, 116.67108073987839]
SFH_VIC_TAC = [1273.574943, 2091.328139, 2909.081337, 3159.696719, 3184.053559, 2284.863062]

elec_vect_NRW = [0.1, 0.2, 0.3, 0.4, 0.5, elec_cons_NRW_ori]
elec_vect_VIC = [0.1, 0.2, 0.3, 0.4, 0.5, elec_cons_VIC_ori]
SFH_NRW_dict = dict(zip(elec_vect_NRW, SFH_NRW_LCOE))
SFH_VIC_dict = dict(zip(elec_vect_VIC, SFH_VIC_LCOE))


###### Database
with open(f'{dirfile_NRW_NoIns[0]}', 'rb') as f: # last generated pickle
    db_NRW_NoIns = pickle.load(f)
with open(f'{dirfile_VIC_NoIns[0]}', 'rb') as f: # last generated pickle
    db_VIC_NoIns = pickle.load(f)
with open(f'{dirfile_NRW_S3[0]}', 'rb') as f: # last generated pickle
    db_NRW_S3 = pickle.load(f)
with open(f'{dirfile_VIC_S3[0]}', 'rb') as f: # last generated pickle
    db_VIC_S3 = pickle.load(f)

###### cleanup of duplicated columns. copy() is used to avoid double slicing in the dataframe
db_NRW_NoIns = db_NRW_NoIns.loc[:,~db_NRW_NoIns.columns.duplicated()].copy()
db_VIC_NoIns = db_VIC_NoIns.loc[:,~db_VIC_NoIns.columns.duplicated()].copy()
db_NRW_S3 = db_NRW_S3.loc[:,~db_NRW_S3.columns.duplicated()].copy()
db_VIC_S3 = db_VIC_S3.loc[:,~db_VIC_S3.columns.duplicated()].copy()

###### Generation of csv files for further assessment in excel
db_NRW_NoIns.to_csv(results_path / file_name_NRW_NoIns / 'database_noins.csv')
db_VIC_NoIns.to_csv(results_path / file_name_VIC_NoIns / 'database_noins.csv')
db_NRW_S3.to_csv(results_path / file_name_NRW_S3 / 'database_s3.csv')
db_VIC_S3.to_csv(results_path / file_name_VIC_S3 / 'database_s3.csv')

###### Database dict
keys = ['NRW_NoIns', 'VIC_NoIns', 'NRW_S3', 'VIC_S3']
db_dict = {'NRW_NoIns': db_NRW_NoIns, 'VIC_NoIns' : db_VIC_NoIns,'NRW_S3': db_NRW_S3, 'VIC_S3' : db_VIC_S3}

######
### Vectors and string list of parameters
######

Tin_vect = [x + 273.15 for x in [5,10,20,30,40,50]] # Temporary HX pinch = 0K
length_vect = [50,100,500,1000,2500,5000,7500,10000] ### 
n_cons_vect = [50,100,500,1000,5000]
elec_vect = [0.1,0.2,0.3,0.4,0.5]
length_str = [str(x) for x in length_vect]
n_cons_str = [str(x) for x in n_cons_vect]
Tin_str = [str(x) for x in Tin_vect]

###### Additional Postprocessing
SFH_VIC_OP_LCOE = []
SFH_VIC_OP_TAC = []
for i,[tac,el_price] in enumerate(zip(SFH_VIC_TAC,elec_vect_VIC)):

    if el_price < 0.32:
        SFH_VIC_OP_TAC.append(tac -  455.82) # values calculated using 01 SFH\ cost_calc.py - only ASHP build
    else:
        SFH_VIC_OP_TAC.append(tac -  (584.085 + 436.95)) # values calculated using 01 SFH\ cost_calc.py - ASHP & Boiler

    SFH_VIC_OP_LCOE.append((SFH_VIC_OP_TAC[i] * SFH_VIC_LCOE[i]) / SFH_VIC_TAC[i])
SFH_VIC_OP_dict = dict(zip(elec_vect_VIC, SFH_VIC_OP_LCOE))


for (k,db) in db_dict.items():
    db['el_price'] =  db['el_price'].apply(lambda x: round(x,4)) # rounding el_price for accurate referencing
    db['LCOE_OP'] = db.apply(lambda row: row['Operation_costs']/row['tot_dem'], axis = 1)
    db['norm_Q'] = db.apply(lambda row: np.round(row['tot_dem']/row['length'],4), axis = 1) # Total demand divided by length
    db['norm_losses'] = db.apply(lambda row: row['total_losses']/row['tot_dem'], axis = 1) # Normalised losses, probably not used 

    if 'NRW' in k:
        db['SFH_LCOE'] = db.apply(lambda row: SFH_NRW_dict[row['el_price']], axis = 1)
    elif 'VIC' in k:
        db['SFH_LCOE'] = db.apply(lambda row: SFH_VIC_dict[row['el_price']], axis = 1)
        db['SFH_LCOE_OP'] = db.apply(lambda row: SFH_VIC_OP_dict[row['el_price']], axis = 1)
        db['comp_LCOE_OP'] = db.apply(lambda row: (row['SFH_LCOE_OP'] - row['LCOE_OP'])/row['SFH_LCOE_OP'], axis = 1) # Total demand divided by length          
    db['comp_LCOE'] = db.apply(lambda row: (row['SFH_LCOE'] - row['LCOE'])/row['SFH_LCOE'], axis = 1) # Total demand divided by length  
    # db['comp_LCOE_OP'] = db.apply(lambda row: (row['SFH_LCOE_OP'] - row['LCOE_OP'])/row['SFH_LCOE_OP'], axis = 1) # Total demand divided by length  
###### Fig list used for plotting and saving figures
fig_list = []



#########
##### FIG 4
#########

##DB Filtering - Base case - Tsource = 283.15 K, elec price = base
db_base_plt = {key: None for key in keys} 
db_base = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_res = db[(db['el_price'] == elec_cons[k]) & (db['T_source'] == (10+273.15))].copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_base[k] = db_res
    db_base_plt[k] = db_res.pivot_table(index='length', columns='n_cons', values='LCOE')

db_base_ele = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_res = db[(db['el_price'] == elec_cons[k])].copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_base_ele[k] = db_res

db_3D = {key: None for key in keys} ## All Variables for 3D analysis 
for (k,db) in db_dict.items():
    db_res = db.copy() #ensure that no double slicing is occuring
    db_res.reset_index(drop = True, inplace=True)
    db_3D[k] = db_res

### Plotting and format
fig4,ax4 = plt.subplots(nrows=1,ncols=2, tight_layout = True, figsize = figsize('medium',60))
ax4[0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax4[1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
db_base_plt['NRW_NoIns'].plot(marker = 'o', markersize = 4, linestyle = '-', ax=ax4[0])
db_base_plt['VIC_NoIns'].plot(marker = 'o', markersize = 4, linestyle = '-', ax=ax4[1])
ax4[0].legend(bbox_to_anchor=(1.0, 1.0))
ax4[0].legend(title = 'No. of consumers')
ax4[1].legend(bbox_to_anchor=(1.0, 1.0))
ax4[1].legend(title = 'No. of consumers')
ax4[0].set(
       ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel="Transmission length [m]")
ax4[1].set( yticklabels = [], 
       xlabel="Transmission length [m]")         
ax4[0].set_ylim([0, 500])
ax4[1].set_ylim([0, 500])
ax4[0].set_xlim([-500, 10500])
ax4[1].set_xlim([-500, 10500])
draw_text(ax4[0], '(a) - NRW', MEDIUM_SIZE, 'upper right')
draw_text(ax4[1],'(b) - VIC', MEDIUM_SIZE, 'upper right')
draw_text(ax4[0], 'Decentralised', SMALL_SIZE, 'lower right')
draw_text(ax4[1], 'Decentralised', SMALL_SIZE, 'lower right')
fig_list.append(fig4)


########
### Further Procesing 
###### Filtering cost breakdown
########

category_labels =  length_str
series_labels = ['Generation Investment', 'Generation Operation',
                'Transmission Investment', 'Transmission Operation',
                'Consumers Investment', 'Consumers Operation',
                'Total Maintenance'
                ]

col_components = ['Generation_Investment','Generation_Maintenance','Generation_Operation',
                'Transmission_Investment','Transmission_Maintenance','Transmission_Operation',
                'Consumers_Investment','Consumers_Maintenance','Consumers_Operation',
                ]
  
col_components_rel = ['Transmission_Investment','Transmission_Operation',
                'Consumers_Operation',
                ]

db_norm_base = {key:None for key in keys}
db_norm_base_costs = {key:None for key in keys}  
dissag_costs = {key:None for key in keys} 
dissag_costs_rel = {key:None for key in keys} 

###### Base costs - Temperature and electricity
####### T source = 10 degC, ele cost = base
####

for (k,db) in db_base.items():
    db_holder = {}
    db_holder_costs = {}
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder[name] = db[db['n_cons'] == n]
        db_holder_costs[name] = db_holder[name].copy() # copying complete DataFrame
        db_holder_costs[name][col_components] = db[db['n_cons'] == n][col_components].div(db[db['n_cons'] == n]['tot_dem'].values,axis=0) # Altering dissagregated costs columns, axis = 0, compare by index
    db_norm_base[k] = db_holder
    db_norm_base_costs[k] = db_holder_costs

db_norm_ele = {key:None for key in keys}
db_norm_ele_costs = {key:None for key in keys}  
dissag_costs = {key:None for key in keys} 
dissag_costs_rel = {key:None for key in keys} 

###### Base costs - Electricity
######## elec cost = base, T source = T_vector
####

for (k,db) in db_base_ele.items():
    db_holder = {}
    db_holder_costs = {}
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder_T = {}
        db_holder_costs_T = {}
        for T,nameT in zip(Tin_vect, Tin_str):
            db_holder_T[nameT] = db[(db['n_cons'] == n) & (db['T_source'] == T)]
            db_holder_costs_T[nameT] = db_holder_T[nameT].copy() # copying complete DataFrame
            for column in [col_components]:
                db_holder_costs_T[nameT][column] = db_holder_costs_T[nameT].apply(lambda row: row[column]/row['tot_dem'], axis = 1) # Altering dissagregated costs columns, axis = 1, compare by index
        db_holder[name] = db_holder_T 
        db_holder_costs[name] = db_holder_costs_T
    db_norm_ele[k] = db_holder
    db_norm_ele_costs[k] = db_holder_costs

#########
### FIG 5
#########

# markers_dict_n = {k:d for (k,d) in zip(n_cons_vect,markers[0:len(n_cons_str)])}
color_dict_n = {k:d for (k,d) in zip(n_cons_vect,colors[0:len(n_cons_str)])}

fig5,ax5 = plt.subplots(1,2,tight_layout = True, figsize = figsize('medium',60))

i = 0
ax_list = []
for db in [db_base.get(k) for k in ['NRW_NoIns','VIC_NoIns']]:
    ax = plt.subplot(1,2,i + 1)
    db_plot_fig5 = db.copy()
    label = f'{{{K2C(T)}}}'
    ax.set_xscale('log')
    ax.set_ylim([0, 500])
    ax.set_xlim([5E-2, 5E3])
    if i == 0:
        ax.fill_between(x_SFH,-10, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set(
            ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
            xlabel="$Q_{\mathrm{tot}}/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")
        draw_text(ax, '(a) - NRW', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Decentralised', SMALL_SIZE, 'lower right')

    else:
        ax.fill_between(x_SFH,-10, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set( yticklabels = [], 
            xlabel="$Q_{\mathrm{tot}}/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")        
        draw_text(ax,'(b) - VIC', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Decentralised', SMALL_SIZE, 'lower right')
    for n_cons,d in db_plot_fig5.groupby('n_cons'):
        ax.scatter(d['norm_Q'],d['LCOE'], c = color_dict_n[n_cons], s = 10, label = str(n_cons))
    i =+ 1
    ax.legend(title = 'No. of \n consumers')
fig_list.append(fig5)



#########
### FIG 6
#########-

color_dict_T = {k:d for (k,d) in zip(Tin_vect,colors[0:len(Tin_vect)])}


#### One plot
fig6,ax6 = plt.subplots(tight_layout = True, figsize = figsize('medium',60), sharex=True, sharey=True)
ax6.remove() #removing overlapping axes
ax_list = []
for i,db in enumerate([db_base_ele.get(k) for k in ['NRW_NoIns','VIC_NoIns']]):
    if ax in locals():
        ax.remove()
    ax = plt.subplot(1,2,i + 1)
    db_plot_fig6 = db.copy()
    text_unit = '$^\circ\mathrm{{C}}$'

    if i == 0:
        ax.fill_between(x_SFH,-10, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set(
            ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
            xlabel="$Q_{\mathrm{tot}}/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")
        draw_text(ax, '(a) - NRW', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Decentralised', SMALL_SIZE, 'lower right')
    else:
        ax.fill_between(x_SFH,-10, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
        ax.set( yticklabels = [], 
        xlabel="$Q_{\mathrm{tot}}/L \: [\mathrm{MWh} \: \mathrm{m}^{-1}]$")        
        draw_text(ax,'(b) - VIC', SMALL_SIZE, 'upper center')
        draw_text(ax, 'Decentralised', SMALL_SIZE, 'lower right')
    ax.set_ylim([0, 500])
    ax.set_xlim([5E-2, 5E3])
    ax.set_xscale('log')
    for T,d in db_plot_fig6.groupby('T_source'):
        T_var = np.round(K2C(T),0)
        ax.scatter(d['norm_Q'],d['LCOE'], s = 5, c = color_dict_T[T], label = f'{T_var}' + text_unit)
    ax.legend(title = '$T_{\mathrm{source}}$', labelspacing = 0.4)
fig_list.append(fig6)

#########
##### FIG 7
#########

# #######
# # Futher processing - TEMP Dependance
# #######
color_dict_L = {k:d for (k,d) in zip([100,1000,5000,10000],colors[0:4])}

db_Tsource_L = {key:None for key in keys}
db_Tsource_L_pvt = {key:None for key in keys}
db_Tsource_n = {key:None for key in keys}
db_Tsource_n_pvt = {key:None for key in keys}

for k,db in db_base_ele.items():
    db['T_source'] = db['T_source'].apply(lambda x: K2C(x)) # To Celsius
    db_holder_L = {}
    db_holder_n = {}
    db_holder_L_pvt = {}
    db_holder_n_pvt = {}   

    for L,name in zip(length_vect,length_str):
        db_holder_L[name] = db[db['length'] == L].copy()
        db_holder_L_pvt[name] =  db_holder_L[name].pivot_table(index='T_source', columns='n_cons', values='LCOE')
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder_n[name] = db[(db['n_cons'] == n) & (db['length'].isin([100,1000,5000,10000]))].copy()
        db_holder_n_pvt[name] =  db_holder_n[name].pivot_table(index='T_source', columns='length', values='LCOE')    

    db_Tsource_L[k] = db_holder_L
    db_Tsource_L_pvt[k] = db_holder_L_pvt
    db_Tsource_n[k] = db_holder_n
    db_Tsource_n_pvt[k] = db_holder_n_pvt

##### 
# N = 100,1000
#NRW
f7,ax7 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_Tsource_n_pvt['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[0,0], color = color_dict_L)
db_Tsource_n_pvt['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[0,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[0,1], color = color_dict_L)
db_Tsource_n_pvt['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[0,1], color = color_dict_L)
db_Tsource_n_pvt['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[1,0], color = color_dict_L)
db_Tsource_n_pvt['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[1,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax7[1,1], color = color_dict_L)
db_Tsource_n_pvt['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax7[1,1], color = color_dict_L)
ax7[0,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[0,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[1,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[1,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax7[0,0].get_legend().remove()
ax7[0,1].get_legend().remove()
ax7[1,0].get_legend().remove()
ax7[1,1].get_legend().remove()


from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=colors[i]) for i in range(4)] ## Custom Legend
patches.reverse() #showing colors at they appear at the plot
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse() #showing symbols at they appear at the plot

l1 = ax7[0,1].legend(handles=patches, labels = [str(x) for x in [10000,5000,1000,100]],bbox_to_anchor=(1.5,1.0), title='Transmission\n length [m]') ##Add 2nd legend
l2 = ax7[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
ax7[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax7[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax7[0,1].set(yticklabels = [],
xlabel = None)

ax7[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='$T_{\mathrm{source}} [^\circ\mathrm{{C}}]$')
ax7[1,1].set(yticklabels = [],
       xlabel='$T_{\mathrm{source}} [^\circ\mathrm{{C}}]$')

ax7[0,0].set_ylim([0, 500])
ax7[0,1].set_ylim([0, 500])
ax7[0,0].set_xlim([0, 55])
ax7[0,1].set_xlim([0, 55])
ax7[1,0].set_ylim([0, 250])
ax7[1,1].set_ylim([0, 250])
ax7[1,0].set_xlim([0, 55])
ax7[1,1].set_xlim([0, 55])
draw_text(ax7[0,0],'(a) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax7[0,0], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax7[0,1],'(b) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax7[0,1], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax7[1,0],'(c) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax7[1,0], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
draw_text(ax7[1,1],'(d) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax7[1,1], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
ax7[0,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[0,1].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[1,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax7[1,1].xaxis.set_major_locator(plt.MultipleLocator(10))
fig_list.append(f7)



#########
##### FIG 8
#########

#####
## Role of insulation in cost breakdown
#####


### All costs
### Extra step required to use barplot function
for (k,db) in db_norm_base.items():
    db_holder = {} # dict value holder
    for n,name in zip(n_cons_vect,n_cons_str):
        norm_costs = [] # list value holder
        for column in db[name][col_components]:
            norm_costs.append(db[name].apply(lambda row: row[column]/row['tot_dem'], axis = 1).tolist())
        db_holder[name] = norm_costs
    dissag_costs[k] = db_holder


#costs that change with length/temperature
for (k,db) in db_norm_base.items():
    db_holder = {} # dict value holder
    for n,name in zip(n_cons_vect,n_cons_str):
        norm_costs = [] # list value holder
        for column in db[name][col_components_rel]:
            norm_costs.append(db[name].apply(lambda row: row[column]/row['tot_dem'], axis = 1).tolist())
        db_holder[name] = norm_costs
    dissag_costs_rel[k] = db_holder
markers = {'o','^','s','P','D','*'}

color_dict_T = {k:d for (k,d) in zip(Tin_vect,colors[0:len(Tin_vect)])}
markers_dict_n = {k:d for (k,d) in zip(Tin_vect,colors[0:len(n_cons_str)])}


### Analysis of cost components relevant for insulation

color_dict_col = {k:d for (k,d) in zip(col_components_rel,colors[0:len(col_components_rel)])}

### For different temperature source
### NRW
f8,ax8 = plt.subplots(nrows=4,ncols=2, tight_layout = True, figsize=figsize('medium',180))
for i,col in enumerate(col_components_rel):
    ax8[0,0].plot(db_norm_ele_costs['NRW_NoIns']['100']['278.15']['length'],db_norm_ele_costs['NRW_NoIns']['100']['278.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[0,0].plot(db_norm_ele_costs['NRW_S3']['100']['278.15']['length'],db_norm_ele_costs['NRW_S3']['100']['278.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[0,1].plot(db_norm_ele_costs['NRW_NoIns']['100']['313.15']['length'],db_norm_ele_costs['NRW_NoIns']['100']['313.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[0,1].plot(db_norm_ele_costs['NRW_S3']['100']['313.15']['length'],db_norm_ele_costs['NRW_S3']['100']['313.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[1,0].plot(db_norm_ele_costs['NRW_NoIns']['1000']['278.15']['length'],db_norm_ele_costs['NRW_NoIns']['1000']['278.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[1,0].plot(db_norm_ele_costs['NRW_S3']['1000']['278.15']['length'],db_norm_ele_costs['NRW_S3']['1000']['278.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[1,1].plot(db_norm_ele_costs['NRW_NoIns']['1000']['313.15']['length'],db_norm_ele_costs['NRW_NoIns']['1000']['313.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[1,1].plot(db_norm_ele_costs['NRW_S3']['1000']['313.15']['length'],db_norm_ele_costs['NRW_S3']['1000']['313.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])    

for i,col in enumerate(col_components_rel):
    ax8[2,0].plot(db_norm_ele_costs['VIC_NoIns']['100']['278.15']['length'],db_norm_ele_costs['VIC_NoIns']['100']['278.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[2,0].plot(db_norm_ele_costs['VIC_S3']['100']['278.15']['length'],db_norm_ele_costs['VIC_S3']['100']['278.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[2,1].plot(db_norm_ele_costs['VIC_NoIns']['100']['313.15']['length'],db_norm_ele_costs['VIC_NoIns']['100']['313.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[2,1].plot(db_norm_ele_costs['VIC_S3']['100']['313.15']['length'],db_norm_ele_costs['VIC_S3']['100']['313.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[3,0].plot(db_norm_ele_costs['VIC_NoIns']['1000']['278.15']['length'],db_norm_ele_costs['VIC_NoIns']['1000']['278.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[3,0].plot(db_norm_ele_costs['VIC_S3']['1000']['278.15']['length'],db_norm_ele_costs['VIC_S3']['1000']['278.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

for i,col in enumerate(col_components_rel):
    ax8[3,1].plot(db_norm_ele_costs['VIC_NoIns']['1000']['313.15']['length'],db_norm_ele_costs['VIC_NoIns']['1000']['313.15'][col], marker = 'o', linestyle='-',fillstyle='none', c = color_dict_col[col])
    ax8[3,1].plot(db_norm_ele_costs['VIC_S3']['1000']['313.15']['length'],db_norm_ele_costs['VIC_S3']['1000']['313.15'][col], marker = '^', linestyle='--',fillstyle='none', c = color_dict_col[col])

ax8[0,0].set_ylim([-25, 250])
ax8[0,1].set_ylim([-25, 250])
ax8[0,0].set_xlim([-500, 10500])
ax8[0,1].set_xlim([-500, 10500])
ax8[1,0].set_ylim([-25, 250])
ax8[1,1].set_ylim([-25, 250])
ax8[1,0].set_xlim([-500, 10500])
ax8[1,1].set_xlim([-500, 10500])
draw_text(ax8[0,0],'(a) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax8[0,0], 'No. Cons = 100,\n$T_\mathrm{{source}} = 10 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[0,1],'(b) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax8[0,1], 'No. Cons = 100,\n$T_\mathrm{{source}} = 40 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[1,0],'(c) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax8[1,0], 'No. Cons = 1000,\n$T_\mathrm{{source}} = 10 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[1,1],'(d) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax8[1,1], 'No. Cons = 1000,\n$T_\mathrm{{source}} = 40 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')

col_components_rel_abbrv = ['Trans. Inv.','Trans. Op.',
                'Cons. Op.',
                ]

ax8[0,0].set(xticklabels = [],
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax8[0,1].set(yticklabels = [], xticklabels = [],
        xlabel=None)

ax8[1,0].set(xticklabels = [],
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax8[1,1].set(yticklabels = [],xticklabels = [],
        xlabel=None)



#### VIC

ax8[2,0].set_ylim([-25, 250])
ax8[2,1].set_ylim([-25, 250])
ax8[2,0].set_xlim([-500, 10500])
ax8[2,1].set_xlim([-500, 10500])
ax8[3,0].set_ylim([-25, 250])
ax8[3,1].set_ylim([-25, 250])
ax8[3,0].set_xlim([-500, 10500])
ax8[3,1].set_xlim([-500, 10500])
draw_text(ax8[2,0],'(e) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax8[2,0], 'No. Cons = 100,\n$T_\mathrm{{source}} = 10 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[2,1],'(f) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax8[2,1], 'No. Cons = 100,\n$T_\mathrm{{source}} = 40 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[3,0],'(g) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax8[3,0], 'No. Cons = 1000,\n$T_\mathrm{{source}} = 10 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')
draw_text(ax8[3,1],'(h) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax8[3,1], 'No. Cons = 1000,\n$T_\mathrm{{source}} = 40 ^\circ\mathrm{{C}}$', SMALL_SIZE, 'upper left')

patches = [mpatches.Patch(color=colors[i]) for i in range(len(col_components_rel))] ## Custom Legend
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse()

l1 = ax8[2,1].legend(handles=patches, labels = [x for x in col_components_rel_abbrv],bbox_to_anchor=(1.0,1.0), loc = 'upper left', title='Cost \n breakdown') ##Add 2nd legend
l2 = ax8[2,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.0,0.4), loc = 'upper left', title='Insulation') ##Add 2nd legend
ax8[2,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax8[2,0].set(xticklabels = [],
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax8[2,1].set(yticklabels = [], xticklabels = [],
    xlabel = None)

ax8[3,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='Transmission Length [m]')
ax8[3,1].set(yticklabels = [],
       xlabel='Transmission Length [m]')

fig_list.append(f8)







#######
### FIG 10
#######

####
# Further processing - Electricity Costs
####

color_dict_el_NRW = {k:d for (k,d) in zip(elec_vect_NRW,colors[0:len(elec_vect_NRW)])}
color_dict_el_VIC = {k:d for (k,d) in zip(elec_vect_VIC,colors[0:len(elec_vect_VIC)])}

##DB Filtering - Base case - Tsource = 283.15 K, elec price = base
db_ele = {key: None for key in keys} 
db_ele_pvt_len = {key: None for key in keys} 
db_ele_pvt_ele = {key: None for key in keys} 
for (k,db) in db_dict.items():
    db_holder_len = {}
    db_holder_ele = {}
    db_holder_pvt_len = {}
    db_holder_pvt_ele = {}
    for n,name in zip(n_cons_vect,n_cons_str):
        db_holder_len[name] = db[(db['T_source'] == (10+273.15)) & (db['n_cons'] == n) & (~db['el_price'].isin([0.2,0.4]))].copy() #ensure that no double slicing is occuring
        db_holder_ele[name] = db[(db['T_source'] == (10+273.15)) & (db['n_cons'] == n) & (db['length'].isin([100,1000,5000,10000]))].copy() #ensure that no double slicing is occuring
        db_holder_pvt_len[name] = db_holder_len[name].pivot_table(index='length', columns='el_price', values='LCOE')
        db_holder_pvt_ele[name] = db_holder_ele[name].pivot_table(index='el_price', columns='length', values='LCOE')
    db_ele[k] = db_holder
    db_ele_pvt_len[k] = db_holder_pvt_len
    db_ele_pvt_ele[k] = db_holder_pvt_ele

f10,ax10 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_ele_pvt_len['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax10[0,0], color = color_dict_el_NRW)
db_ele_pvt_len['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax10[0,0], color = color_dict_el_NRW)
db_ele_pvt_len['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax10[0,1], color = color_dict_el_VIC)
db_ele_pvt_len['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax10[0,1], color = color_dict_el_VIC)
db_ele_pvt_len['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax10[1,0], color = color_dict_el_NRW)
db_ele_pvt_len['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax10[1,0], color = color_dict_el_NRW)
db_ele_pvt_len['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax10[1,1], color = color_dict_el_VIC)
db_ele_pvt_len['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax10[1,1], color = color_dict_el_VIC)
# ax10[0,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
# ax10[0,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
# ax10[1,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
# ax10[1,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax10[0,0].get_legend().remove()
ax10[0,1].get_legend().remove()
ax10[1,0].get_legend().remove()
ax10[1,1].get_legend().remove()

patches = [mpatches.Patch(color=colors[i]) for i in [4,2,0,5]] ## Custom Legend
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse() #showing colors at they appear at the plot

l1 = ax10[0,1].legend(handles=patches, labels = ['0.5','0.3','0.1','base price\nNRW: 0.32\nVIC: 0.22'],bbox_to_anchor=(1.5,1.1), title='El. price \n $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$') ##Add 2nd legend
l2 = ax10[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
ax10[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax10[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax10[0,1].set( yticklabels = [],
xlabel = None)

ax10[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='Transmission Length [m]')
ax10[1,1].set( yticklabels = [],
       xlabel='Transmission Length [m]')

ax10[0,0].set_ylim([0, 500])
ax10[0,1].set_ylim([0, 500])
ax10[0,0].set_xlim([-500, 10500])
ax10[0,1].set_xlim([-500, 10500])
ax10[1,0].set_ylim([0, 500])
ax10[1,1].set_ylim([0, 500])
ax10[1,0].set_xlim([-500, 10500])
ax10[1,1].set_xlim([-500, 10500])
draw_text(ax10[0,0],'(a) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax10[0,0], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax10[0,1],'(b) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax10[0,1], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax10[1,0],'(c) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax10[1,0], 'No. Cons = 1000', SMALL_SIZE, 'upper left')
draw_text(ax10[1,1],'(d) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax10[1,1], 'No. Cons = 1000', SMALL_SIZE, 'upper left')

fig_list.append(f10)

###
## FIG 11 - el_price
###
el_price_color = [0.1, 0.3, 0.4, 0.5,0.7]
LCOE_NRW_color = [72.87455876557871, 163.4664062682543,168.3000122546595,168.4452107587107,168.4452107587107]
LCOE_VIC_color = [65.03206581590275, 148.5452975719377, 161.3423741098446, 162.58609806912528, 162.58609806912528]

f11,ax11 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_ele_pvt_ele['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[0,0], color = color_dict_L)
db_ele_pvt_ele['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[0,0], color = color_dict_L)
db_ele_pvt_ele['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[0,1], color = color_dict_L)
db_ele_pvt_ele['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[0,1], color = color_dict_L)
db_ele_pvt_ele['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[1,0], color = color_dict_L)
db_ele_pvt_ele['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[1,0], color = color_dict_L)
db_ele_pvt_ele['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax11[1,1], color = color_dict_L)
db_ele_pvt_ele['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax11[1,1], color = color_dict_L)
ax11[0,0].fill_between(el_price_color,0, LCOE_NRW_color, color = 'gray', alpha = 0.6)
ax11[0,1].fill_between(el_price_color,0, LCOE_VIC_color, color = 'gray', alpha = 0.6)
ax11[1,0].fill_between(el_price_color,0, LCOE_NRW_color, color = 'gray', alpha = 0.6)
ax11[1,1].fill_between(el_price_color,0, LCOE_VIC_color, color = 'gray', alpha = 0.6)
ax11[0,0].get_legend().remove()
ax11[0,1].get_legend().remove()
ax11[1,0].get_legend().remove()
ax11[1,1].get_legend().remove()

patches = [mpatches.Patch(color=colors[i]) for i in range(4)] ## Custom Legend
patches.reverse()
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse()

l1 = ax11[0,1].legend(handles=patches, labels = [str(x) for x in [10000,5000,1000,100]],bbox_to_anchor=(1.5,1.0), title='Transmission\n length [m]') ##Add 2nd legend
l2 = ax11[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
ax11[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax11[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax11[0,1].set(yticklabels = [],
                xlabel = None)

ax11[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')
ax11[1,1].set(yticklabels = [],
       xlabel='Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')

ax11[0,0].set_ylim([0, 500])
ax11[0,1].set_ylim([0, 500])
ax11[0,0].set_xlim([0.05, 0.55])
ax11[0,1].set_xlim([0.05, 0.55])
ax11[1,0].set_ylim([0, 500])
ax11[1,1].set_ylim([0, 500])
ax11[1,0].set_xlim([0.05, 0.55])
ax11[1,1].set_xlim([0.05, 0.55])
draw_text(ax11[0,0],'(a) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax11[0,0], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax11[0,1],'(b) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax11[0,1], 'No. Cons = 100', SMALL_SIZE, 'upper left')
draw_text(ax11[1,0],'(c) - NRW', SMALL_SIZE, 'upper right')
draw_text(ax11[1,0], 'No. Cons = 1000', SMALL_SIZE, 'upper left')
draw_text(ax11[1,1],'(d) - VIC', SMALL_SIZE, 'upper right')
draw_text(ax11[1,1], 'No. Cons = 1000', SMALL_SIZE, 'upper left')
fig_list.append(f11)

#######
## Extra FIG - RUM Conference - No Insulation
#######

##### 
# N = 100,1000
#NRW
f12,ax12 = plt.subplots(nrows=2,ncols=2, tight_layout = True, figsize=figsize('medium',90))
db_Tsource_n_pvt['NRW_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax12[0,0], color = color_dict_L)
# db_Tsource_n_pvt['NRW_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax12[0,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['100'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax12[0,1], color = color_dict_L)
# db_Tsource_n_pvt['VIC_S3']['100'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax12[0,1], color = color_dict_L)
db_Tsource_n_pvt['NRW_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax12[1,0], color = color_dict_L)
# db_Tsource_n_pvt['NRW_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax12[1,0], color = color_dict_L)
db_Tsource_n_pvt['VIC_NoIns']['1000'].plot(marker = 'o', linestyle='-',fillstyle='none', ax=ax12[1,1], color = color_dict_L)
# db_Tsource_n_pvt['VIC_S3']['1000'].plot(marker = '^', linestyle='--',fillstyle='none', ax=ax12[1,1], color = color_dict_L)
ax12[0,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax12[0,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax12[1,0].fill_between(x_SFH,0, SFH_NRW_LCOE[-1], color = 'gray', alpha = 0.6)
ax12[1,1].fill_between(x_SFH,0, SFH_VIC_LCOE[-1], color = 'gray', alpha = 0.6)
ax12[0,0].get_legend().remove()
ax12[0,1].get_legend().remove()
ax12[1,0].get_legend().remove()
ax12[1,1].get_legend().remove()


from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=colors[i]) for i in range(4)] ## Custom Legend
patches.reverse() #showing colors at they appear at the plot
myHandle = [Line2D([], [], marker='o', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='-', color = 'k'),
          Line2D([], [], marker='^', markerfacecolor='None', markeredgecolor='k', markersize=10, linestyle='--', color = 'k')] ##Create custom handles for 2nd legend
myHandle.reverse() #showing symbols at they appear at the plot

l1 = ax12[0,1].legend(handles=patches, labels = [str(x) for x in [10000,5000,1000,100]],bbox_to_anchor=(1.5,1.0), title='Transmission\n length [m]') ##Add 2nd legend
# l2 = ax12[0,1].legend(handles=myHandle, labels = ['Ins S3','No Ins'],bbox_to_anchor=(1.5,0.3), title='Insulation') ##Add 2nd legend
# ax12[0,1].add_artist(l1) # 2nd legend will erases the first, so need to add it

ax12[0,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
        xlabel=None)
ax12[0,1].set(yticklabels = [],
xlabel = None)

ax12[1,0].set(
        ylabel="LCOE $[\mathrm{EUR} \: \mathrm{MWh}^{-1}]$",
       xlabel='$T_{\mathrm{source}} [^\circ\mathrm{{C}}]$')
ax12[1,1].set(yticklabels = [],
       xlabel='$T_{\mathrm{source}} [^\circ\mathrm{{C}}]$')

ax12[0,0].set_ylim([0, 500])
ax12[0,1].set_ylim([0, 500])
ax12[0,0].set_xlim([0, 55])
ax12[0,1].set_xlim([0, 55])
ax12[1,0].set_ylim([0, 500])
ax12[1,1].set_ylim([0, 500])
ax12[1,0].set_xlim([0, 55])
ax12[1,1].set_xlim([0, 55])
draw_text(ax12[0,0],'(a) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax12[0,0], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax12[0,1],'(b) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax12[0,1], 'No. Cons = 100', SMALL_SIZE, 'lower left')
draw_text(ax12[1,0],'(c) - NRW', SMALL_SIZE, 'upper center')
draw_text(ax12[1,0], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
draw_text(ax12[1,1],'(d) - VIC', SMALL_SIZE, 'upper center')
draw_text(ax12[1,1], 'No. Cons = 1000', SMALL_SIZE, 'lower left')
ax12[0,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax12[0,1].xaxis.set_major_locator(plt.MultipleLocator(10))
ax12[1,0].xaxis.set_major_locator(plt.MultipleLocator(10))
ax12[1,1].xaxis.set_major_locator(plt.MultipleLocator(10))
fig_list.append(f12)

# additional VIC figure
fig_list.append(f12)
##PLOT
for i,fig in enumerate(fig_list):
    # if i in []:
    #     pass
    fig.savefig(f'Fig_{i+6}_Decen_rev2.pdf', dpi=1000)
    # else:
        # plt.close(fig)


# plt.show()

#########
### FIG 13
#########-



color_dict_T = {k:d for (k,d) in zip(Tin_vect,colors[0:len(Tin_vect)])}
color_dict_T = {k:d for (k,d) in zip(Tin_vect,colors[0:len(Tin_vect)])}

#### One plot
import scipy, scipy.optimize
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm # to colormap 3D surfaces from blue to red
import matplotlib.pyplot as plt
# fig13,ax13 = plt.subplots(tight_layout = True, figsize = figsize('medium',60), sharex=True, sharey=True)
# ax13.remove() #removing overlapping axes
# ax_list = []
# ax13 = plt.axes(projection='3d')
# for i,db in enumerate([db_3D.get(k) for k in ['NRW_NoIns','VIC_NoIns']]):

def SurfacePlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)


    axes = plt.axes(projection='3d')
    matplotlib.pyplot.grid(True)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = np.linspace(min(x_data), max(x_data), 20)
    yModel = np.linspace(min(y_data), max(y_data), 20)
    X, Y = np.meshgrid(xModel, yModel)

    Z = func(np.array([X, Y]), *fittedParameters)

    axes.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=1, antialiased=True)

    axes.scatter3D(x_data, y_data, z_data) # show data along with plotted surface

    axes.set_title('Surface Plot (click-drag with mouse) - Adjustment') # add a title for surface plot
    axes.set_xlabel('$Q_{tot}/L [MWh/m]$')
    axes.set_ylabel('el. prices [cEUR/kWh]')
    axes.set_zlabel('norm. LCOE')

    plt.show()
    plt.close('all') # clean up after using pyplot or else there can be memory and process problems


def ContourPlot(func, data, fittedParameters):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = f.add_subplot(111)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    xModel = np.linspace(min(x_data), max(x_data), 20)
    yModel = np.linspace(min(y_data), max(y_data), 20)
    X, Y = np.meshgrid(xModel, yModel)

    Z = func(np.array([X, Y]), *fittedParameters)

    axes.plot(x_data, y_data, 'o')

    axes.set_title('Contour Plot') # add a title for contour plot
    axes.set_xlabel('X Data') # X axis data label
    axes.set_ylabel('Y Data') # Y axis data label

    CS = matplotlib.pyplot.contour(X, Y, Z, numberOfContourLines, colors='k')
    matplotlib.pyplot.clabel(CS, inline=1, fontsize=10) # labels for contours

    plt.show()
    plt.close('all') # clean up after using pyplot or else there can be memory and process problems


def ScatterPlot(data):
    f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
    axes = plt.axes(projection='3d')
    matplotlib.pyplot.grid(True)

    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    axes.scatter3D(x_data, y_data, z_data)

    axes.set_title('Scatter Plot (click-drag with mouse) - Data')
    axes.set_xlabel('$Q_{tot}/L [MWh/m]$')
    axes.set_ylabel('el. prices [cEUR/kWh]')
    axes.set_zlabel('norm. LCOE')

    plt.show()
    plt.close('all') # clean up after using pyplot or else there can be memory and process problems

# def func(data, a, b,c,d,e):
#     x = data[0] #exp form, a*np.exp(b*x)
#     y = data[1] #quad_form, c*y**2 + d*y
#                 # plus constant e
#     return a*np.exp(b*x) + c*y**2+d*y+e


def func(data, a, b, c):
    x = data[0]
    y = data[1]
    return (a * x) + (y * b) + c

graphWidth = 800 # units are pixels
graphHeight = 600 # units are pixels

# 3D contour plot lines
numberOfContourLines = 16

# for i,db in enumerate([db_3D.get(k) for k in ['VIC_NoIns']]):
#     if ax in locals():
#         ax.remove()
#     # ax = plt.subplot(1,1,1)
#     # ax = plt.axes(projection='3d')
#     # db_plot_fig13 = db.copy()
#     # text_unit = '$^\circ\mathrm{{C}}$'
#     # ax.set_zlim([-0.5, 0.5])
#     # ax.set_xlim([5E-2, 5E3])
#     # ax.set_xscale('log')
#     # for T,d in db_plot_fig13.groupby('el_price'):
#     bins = [1E-2, 1E-1, 1E0, 1E1, 1E2, 1E3, 1E4]
#     i_color = 0
#     db_pivot = db[(db['T_source'] == (10+273.15)) & (db['norm_Q'] >= 0.0)] ## no longer a pivot, just a variable name
#     # db_pivot = pd.pivot_table(db[(db['T_source'] == (10+273.15)) & (db['comp_LCOE'] >= 0.0)], index = ['norm_Q','el_price'])
#     # db_pivot = pd.pivot_table(db[(db['norm_Q'] >= 10.0)], index = ['norm_Q','el_price'])
#     # X_pd,Y_pd= map(np.array, zip(*db_pivot.index.values))
#     X_pd = db_pivot['norm_Q'].apply(np.log10)
#     # X_pd = db_pivot['norm_Q']
#     Y_pd = db_pivot['el_price']
#     inpData = np.column_stack((X_pd,Y_pd))
#     LCOE_Data = db_pivot['comp_LCOE'].values
#     INV_Data = db_pivot.apply(lambda row: row['Investment_costs'], axis=1).values
#     MAIN_Data = db_pivot.apply(lambda row: row['Maintenance_costs'], axis=1).values
#     OP_Data = db_pivot.apply(lambda row: row['Operation_costs']/row['tot_dem'], axis=1).values
#     # INV_Data = db_pivot.apply(lambda row: row['Investment_costs']/row['tot_dem'], axis=1).values
#     # MAIN_Data = db_pivot.apply(lambda row: row['Maintenance_costs']/row['tot_dem'], axis=1).values
#     # OP_Data = db_pivot.apply(lambda row: row['Operation_costs']/row['tot_dem'], axis=1).values
#     data = [X_pd, Y_pd, LCOE_Data]
#     initialParameters = [1.0, 1.0, 1.0] # these are the same as scipy default values in this example

#     # here a non-linear surface fit is made with scipy's curve_fit()
#     fittedParameters, pcov = scipy.optimize.curve_fit(func, [X_pd, Y_pd], LCOE_Data, p0 = initialParameters)
#     # ScatterPlot([db_pivot['n_cons'],db_pivot['length'],db_pivot['LCOE']])

#     ScatterPlot(data)
#     SurfacePlot(func, data, fittedParameters)
#     print('fitted prameters', fittedParameters)
#     modelPredictions = func(data, *fittedParameters) 
#     absError = modelPredictions - LCOE_Data
#     SE = np.square(absError) # squared errors
#     MSE = np.mean(SE) # mean squared errors
#     RMSE = np.sqrt(MSE) # Root Mean Squared Error, RMSE
#     Rsquared = 1.0 - (np.var(absError) / np.var(LCOE_Data))
#     print('RMSE:', RMSE)
#     print('R-squared:', Rsquared)

# def func2D(data, a, b, c, d,e):
#     x = data[0]
#     # y = data[1]
#     return (e * x**4) + (d * x**3) + (a * x**2) + (x * b) + c
#     # return (a * x**3) + (b * x**2) + (c * x) + d

# #norm Q values assigned to colors for plotting


fig15,ax15 = plt.subplots(tight_layout = True, figsize = figsize('large',120))
ax15.remove() #removing overlapping axes
sorted_elec = elec_vect.copy()
sorted_elec.sort()

import palettable
import matplotlib as mpl
cm = mpl.colors.ListedColormap(palettable.scientific.diverging.Berlin_5.mpl_colors)
offset = lambda p: mpl.transforms.ScaledTranslation(p/72.,0, plt.gcf().dpi_scale_trans)

def Q_color_eval_HEX(Q):
    if Q < 1:
        return colors[-1]
    elif Q < 5:
        return colors[0]
    elif Q < 10:
        return colors[1]
    elif Q < 50:
        return colors[2]
    else:
        return colors[3]

def Q_color_eval(Q):
    if Q < 1:
        return 0.5
    # elif Q < 5:
    #     return 1.5
    elif Q < 5:
        return 1.5
    elif Q < 10:
        return 2.5
    elif Q < 50:
        return 3.5
    else:
        return 4.5

def Q_zorder(Q):
    if Q < 1:
        return 1
    # elif Q < 5:
    #     return 1.5
    elif Q < 5:
        return 2
    elif Q < 10:
        return 3
    elif Q < 50:
        return 4
    else:
        return 5
    
def Q_offset(Q):
    if Q < 1:
        return -8.0
    elif Q < 5:
        return -4.0
    elif Q < 10:
        return 0.0
    elif Q < 50:
        return 4.0
    else:
        return 8.0



# print(color_dict_Q)
# for j,db in enumerate([db_3D.get(k) for k in ['VIC_NoIns','VIC_S3']]):

for j,(db_n,db_s) in enumerate(zip([db_3D.get(k) for k in ['VIC_NoIns','NRW_NoIns']],[db_3D.get(k) for k in ['VIC_S3','NRW_S3']])):
    Q_array = db_n['norm_Q'].values
    Q_list = np.unique(np.round(Q_array,4)).tolist()
    color_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_color_eval_HEX(q) for q in Q_list])}
    zorder_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_zorder(q) for q in Q_list])}
    offset_dict_Q = {k:d for (k,d) in zip(Q_list,[Q_offset(q) for q in Q_list])}
    fit_param = { }
    counter_f = 0
    def sep_same_Q(grp):
        global counter_f
        grp['group'] = counter_f
        counter_f += 1
        return grp

    def assign_val(grp): # assign group based on norm_Q, requires a global counter
        global counter_f
        if grp['count'].iloc[0] != 36: ## in cases where more than 1 set fit the same Q value, they are identified first
            grp = grp.groupby(['len_con'],group_keys = False).apply(sep_same_Q) # then an additional separator is implemented based on the length, consumer combination.
    
        else:
            grp['group'] = counter_f
            counter_f += 1
        return grp
    for db in [db_n,db_s]:
        db['len_con'] = db.apply(lambda row: str(row['length']) + ',' + str(row['n_cons']), axis = 1)
        db['count'] = db.groupby(['norm_Q'],group_keys = False)['norm_Q'].transform('count')
        db = db.groupby(['norm_Q'], group_keys = False).apply(assign_val)
        db.sort_values(by=['group'], inplace=True)
        counter_f = 0
    minmax_dict = dict.fromkeys(sorted_elec)
    short_T_vect = [x +273.15 for x in [10, 30, 50]]
    for i,T in enumerate(short_T_vect):
        if ax in locals():
            ax.remove()
        minmax_dict = dict.fromkeys(sorted_elec)
        ax = plt.subplot(2,3,(j)*3 + (i+1))
        for (g_n,d_n),(g_s,d_s) in zip(db_n.groupby(db['group']),db_s.groupby(db['group'])):
            Q = np.unique(d_n['norm_Q'].to_numpy())
            if Q < 1.0:
                pass
            else:
                d_fit_n = d_n[(d_n['T_source'] == T) & (d_n['el_price'].isin(elec_vect))].copy()
                d_fit_s = d_s[(d_s['T_source'] == T) & (d_s['el_price'].isin(elec_vect))].copy()
                col_s_name = {}
                for name in d_fit_s.columns:
                    col_s_name[name] = name + '_s'
                d_fit_s.rename(columns = col_s_name, inplace=True)
                ### concat the dataframes and use row
                conc_d = pd.concat([d_fit_n,d_fit_s],axis = 1)
                conc_d['min_LCOE'] = conc_d.apply(lambda row: row['comp_LCOE'] if row['comp_LCOE']<row['comp_LCOE_s'] else row['comp_LCOE_s'], axis = 1)
                conc_d['marker'] = conc_d.apply(lambda row: 'o' if row['comp_LCOE']<row['comp_LCOE_s'] else '^', axis = 1) ## check marker
                X_pd = conc_d['el_price'].values
                LCOE_Data = conc_d['min_LCOE'].values
                c_pd = []
                for i,c in enumerate(X_pd):
                    c_pd.append(color_dict_Q[Q[0]]) ## accessing the single value array and applying color function 
                m_pd = conc_d['marker'].values
                trans = plt.gca().transData
                for _m, _c, _x, _y in zip(m_pd, c_pd, X_pd, LCOE_Data):
                    # # plt.scatter(_x,_y, s=20, c=_c, vmin=0, vmax=5, cmap=cm, marker=_m, zorder = zorder_dict_Q[Q[0]], ## C-Map version
                    # #                             alpha = 0.6, transform = trans + offset(offset_dict_Q[Q[0]]))
                    plt.scatter(_x,_y, s=20, marker=_m, edgecolors = _c, zorder = zorder_dict_Q[Q[0]], ## no c-map
                                                alpha = 0.6, facecolors = 'none', transform = trans + offset(offset_dict_Q[Q[0]]))
                for x,y in zip(X_pd, LCOE_Data): ## overall band
                    if minmax_dict[x] == None: # initialising dict values, originally empty
                        minmax_dict[x] = [y,y] # [min,max]
                    if y < minmax_dict[x][0]: #min
                        minmax_dict[x][0] = y
                    if y > minmax_dict[x][1]: #max
                        minmax_dict[x][1] = y
        np_minmax = np.array([v for v in minmax_dict.values()])
        plt.axhline(y=0.0, color='black', linestyle='--')
        x_long = np.linspace(sorted_elec[0],sorted_elec[-1],50000)
        y_long_min = np.interp(x_long,sorted_elec,np_minmax[:,0])
        y_long_max = np.interp(x_long,sorted_elec,np_minmax[:,1])
        ax.fill_between(x_long, y_long_min, y_long_max ,where=y_long_max<= 0, color = 'darkkhaki', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.fill_between(x_long, y_long_min, 0 ,where=((y_long_max>=0) & (y_long_min<=0)), color = 'darkkhaki', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.fill_between(x_long, y_long_min, y_long_max,where=((y_long_max>=0) & (y_long_min>=0)), color = 'gray', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE       
        ax.fill_between(x_long,0, y_long_max,where=((y_long_max>=0) & (y_long_min<=0)), color = 'gray', alpha = 0.6, interpolate = True, ec = 'none')       # if LCOE
        ax.set_ylim([-0.45,0.45])
        ax.set_xlim([0.05, 0.55])
        ax.set_xticks(elec_vect, labels = [str(x) for x in elec_vect])
        ax.set_yticks([-0.4, 0, 0.4], labels = [str(x) for x in [-0.4, 0, 0.4]])

    fig15.supxlabel('Electricity Price $[\mathrm{EUR} \: \mathrm{kWh}^{-1}]$')
    # fig15.supylabel(r'Normalised LCOE: $\frac{LCOE_{SFH} - LCOE_{system}}{LCOE_{SFH}} [-]$')
    fig15.supylabel(r'Normalised LCOE $\: [-]$')
def remove_inner_ticklabels(fig):
    for ax in fig.axes:
        try:
            ax.label_outer()
        except:
            pass
remove_inner_ticklabels(fig15)
fig15.savefig(f'Fig_5_Decen_rev2_color_mod.pdf', dpi=1000)
# plt.show() 
