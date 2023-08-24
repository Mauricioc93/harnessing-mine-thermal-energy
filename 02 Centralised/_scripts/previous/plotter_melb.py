#overall packages
import pandas as pd
import pickle
import sys
import os
import glob
import numpy as np

### Set file directory as working dir.
os.chdir(sys.path[0])

### Plot characteristics
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

###### Scientific Plot
# print(plt.style.available)
plt.style.use(['science','nature'])

##### General
# plt.rcParams.update({
#     "font.family": "sans-serif",  # use serif/main font for text elements
#     "font.sans-serif": ["Helvetica"], # use Helvetica
#     "text.usetex": False,     # use inline math for ticks
#     "pgf.rcfonts": False,    # don't setup fonts from rc parameters
#     'hatch.linewidth': 0.5   # reduce hatch linewidth
#     })

S_SIZE = 8 # Small 
M_SIZE = 9 # Medium
L_SIZE = 12 # Large

plt.rc('font', size=S_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=M_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=M_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=M_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=M_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=M_SIZE)    # legend fontsize
plt.rc('figure', titlesize=L_SIZE)  # fontsize of the figure title

####### Time series

month_year_formatter = mdates.DateFormatter('%b') # The "," is intentional.
monthly_locator = mdates.MonthLocator()
### Load results

###### list of files
dirfiles = [f for f in glob.glob("*_melb_results.pickle")]

###### Pickle
with open(f'{dirfiles[-1]}', 'rb') as f: # last generated pickle
    P = pickle.load(f)

###### Problem variable reading
params = P.data.getter()
operation = P.operation.transpose()
design = P.design.transpose()
idx = pd.to_datetime(operation.index.to_series()) # to date

temp = params['HP_C_heat_consumer_T_amb'].apply(lambda x: x-273.15)
heat_demand = params['BES_C_heat_consumer_heating_Qdot'].apply(lambda x: np.nan if x==0 else x)
cool_demand = params['BES_C_heat_consumer_cooling_Qdot'].apply(lambda x: x*-1 if x>0 else np.nan)

#Energy Provision
HP_H = operation.HP_C_heat_consumer_qdot_b_h
ELE_H = operation.apply(lambda row:  row['HS_el_C_heat_consumer_q_dot'] + row['HP_C_heat_consumer_qdot_b_h'] if row['HS_el_C_heat_consumer_q_dot'] > 0 else np.nan, axis = 1)
print(sum(ELE_H))
# COP
COP_H = operation.apply(lambda row: np.nan if row['HP_C_heat_consumer_qdot_b_h'] == 0 else row['HP_C_heat_consumer_qdot_b_h']/row['HP_C_heat_consumer_P_HP_h'], axis = 1)
COP_C = operation.apply(lambda row: np.nan if row['HP_C_heat_consumer_qdot_b_c'] == 0 else row['HP_C_heat_consumer_qdot_b_c']/row['HP_C_heat_consumer_P_HP_c'], axis = 1)


### COP verification
COP_H_ver = temp.apply(lambda t: (45+273.15)/(45-(t-5))*0.5 if 45-(t-5) != 0 else 0).replace({0:pd.NA})
COP_C_ver = temp.apply(lambda t: 0  if (t+5)-7 == 0 else (7+273.15)/((t+5)-7)*0.4).replace({0:pd.NA})

### Figures
def single_plot(x, y, ax=None, plt_kwargs = {}):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **plt_kwargs) ## example plot here

    return(ax)

def multiple_custom_plots(x, y, ax=None, temp_kwargs={}, cop_kwargs={}):
    if ax is None:
        ax = plt.gca()
    ax.plot(x, y, **temp_kwargs) #example temp
    ax.scatter(x, y, **cop_kwargs) #example cop
    return(ax)

###### plot properties

temp_kwargs = {'linewidth': 1.0, 'c': 'gray'}
heat_kwargs = {'linewidth': 1.0, 'c': 'red'}
cool_kwargs = {'linewidth': 1.0, 'c': 'blue'}
COP_kwargs = {'linewidth': 0.8, 'c': 'green'}
COP_kwargs_ver = {'linewidth': 0.8, 'c': 'purple', 'linestyle':'--'}

HP_H_kwargs = {'linewidth': 1.0, 'c': 'green'}
ELE_H_kwargs = {'linewidth': 1.0, 'c': 'yellow'}

###### plot
fig, axes = plt.subplots(4, sharex = True, tight_layout = True, figsize = (10,6.5))
single_plot(idx,temp,ax=axes[0],plt_kwargs = temp_kwargs)
single_plot(idx,heat_demand,ax=axes[1],plt_kwargs = heat_kwargs)
single_plot(idx,cool_demand,ax=axes[1],plt_kwargs = cool_kwargs)
single_plot(idx,COP_H,ax=axes[2],plt_kwargs = COP_kwargs)
# single_plot(idx,COP_H_ver,ax=axes[2],plt_kwargs = COP_kwargs_ver)

single_plot(idx,ELE_H,ax=axes[3],plt_kwargs = ELE_H_kwargs)
single_plot(idx,HP_H,ax=axes[3],plt_kwargs = HP_H_kwargs)

###### formatting

axes[0].set(title ="VIC - Single family House",
       ylabel="Dry Bulb \n Temperature [°C]")         
axes[0].set_ylim([-10, 40])

axes[1].set(ylabel="Heating/Cooling \n Demand [kWh]")
axes[1].legend(['Heating load'])
axes[1].set_ylim([-10, 10])

axes[2].set(ylabel="COP_H [-]")
axes[2].legend(['ASHP'])
axes[2].set_ylim([0, 8])

axes[2].set(xlabel="Date - TMY",
       ylabel="Heat Provisioning [kW]")
axes[2].legend(['HP Load'])
axes[2].set_ylim([0, 10])


# fig.autofmt_xdate()
fig.align_ylabels()
axes[-1].xaxis.set_major_locator(monthly_locator)
axes[-1].xaxis.set_major_formatter(month_year_formatter)
plt.minorticks_off()
plt.show()

# fig.savefig('Fig_test.svg', format='svg', dpi=300)


