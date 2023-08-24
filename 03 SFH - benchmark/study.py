'''Decen - case study for VIC & NRW'''
# AUTHOR: Mauricio Carcamo M.

#### file - database handling and housekeeping
import os
from pathlib import Path

### Run definitions

region = 'VIC' # NRW or VIC
num_typ_days = 24 ## Benchmark 4,12,24,36,48,72
if num_typ_days == None:
    typ_day_file = None
else:
    typ_day_file = region + '_Medoid_' + str(num_typ_days) + 'd_norescale.csv'

run = region + '_SFH' + '_detailed_v3'

reprocessing = True ### if reprocessing a existing dataset, reprocessing = True, otherwise False
repro_name_list = ['VIC_SFH_detailed_v3'] ## if empty, no reprocessing will occur

# Decentralised case

#### script import

parent_path = Path(__file__).parents[0]
os.chdir(str(parent_path))



#### Utilities functions

def find_neighbours(value, df, colname):
    exactmatch = df[df[colname] == value]
    if not exactmatch.empty:
        return exactmatch.index
    else:
        upperneighbour_ind = df[df[colname] > value][colname].idxmin()
        return upperneighbour_ind
i_count = 1

def DN_calc(dT,T,q_c,q_h,n_con,pipeset, cooling = 'yes'):

    #heating
    cp = 4.184
    eff = 0.5
    Th_cold = T
    Th_hot = 45+273.15
    COP_H =Th_hot/(Th_hot-Th_cold)*eff
    m_max_h = q_h/(dT*cp)*(1-1/COP_H)*1.1*n_con

    #cooling

    
    Tc_cold = 7 + 273.15
    Tc_hot = T
    COP_H =Tc_cold/(Tc_hot-Tc_cold)*eff
    m_max_c = q_c/(dT*cp)*(1+1/COP_H)*1.1*n_con
    if cooling == 'yes':
        m_max_tot = max(m_max_c,m_max_h)
    else:
        m_max_tot = m_max_h
    DN_ind = find_neighbours(m_max_tot,pipeset,'Max_mflow')
    return pipeset['DN'].iloc[DN_ind]


import time
start = time.time()

def calc_func(pset): #function for model execution, folder creation and data storage

    import pickle
    import pandas as pd
    import numpy as np
    from _scripts.utilities import annuity
    # import _scripts.pyomoio as po
    from pyomo.environ import value

    print('Simulation running %10.2f min' %((time.time()-start)/60))
    print('ncon = %5d' %pset['n_cons'])


    targetdir = parent_path / pset['_calc_dir'] / pset['_pset_id']
    targetdir.mkdir(parents=True,exist_ok=True)
    pickledir = targetdir / str('results' + '.pk')

    if pset['typical_days'] == 'None': # to pass None argument to run_model function)
        typical_days = None
    else:
        typical_days = pset['typical_days']

    P,m = run_model(ES,operational_calc=False, case = region, typ_days = typical_days, interface = 'pyomo',**pset) ## Remember to instantiate model
    tac = value(m.objective)
    h_demands = sum(p for p in P.parameters if p.name.endswith('heating_Qdot'))
    c_demands = sum(p for p in P.parameters if p.name.endswith('cooling_Qdot'))
    tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3
 
    with pickledir.open('wb') as fp:
        pickle.dump((P), fp)

    P.design.transpose().to_csv(targetdir / 'design_var.csv')
    P.operation.transpose().to_csv(targetdir / 'operation_var.csv')
    P.data.getter().to_csv(targetdir / 'parameters_inp.csv')
    del P, m
    print('LCOE: ' + str(tac/tot_demand))

    return {'tac': tac,
     'LCOE':tac/tot_demand,
    }

 
def post_process_cost(pset,run): #function for post processing of the costs - not used for SFH
    import pickle
    import pandas as pd
    import numpy as np
    from _scripts.utilities import annuity
    # import _scripts.pyomoio as po
    from pyomo.environ import value
    calc_dirs_list = [x for x in pset['_calc_dir']]
    pset_id_list = [x for x in pset['_pset_id']]
    typ_days_list = [x for x in pset['typical_days']]
    Dic_costs = {}

    for typ_days,c_dir,pset_id in zip(typ_days_list,calc_dirs_list,pset_id_list):
        targetdir = parent_path / c_dir / pset_id
        pickledir = targetdir / str('results' + '.pk')
        with open(f'{pickledir}', 'rb') as f: # results pickle
            P = pickle.load(f)

        ### Costs
        params = P.data.getter()
        operation = P.operation.transpose()
        design = P.design.transpose()
        
        comp_list = ['HS_gas','HS_el','Pipes','ASHP','HP_LC','HX_LC','Cen_HX'] ## update LC to HX eventually
        h_demands = sum(p for p in P.parameters if p.name.endswith('heating_Qdot'))
        c_demands = sum(p for p in P.parameters if p.name.endswith('cooling_Qdot'))
        tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3
        heat_tot_demand = P.weighted_sum(h_demands, symbolic=False) / 1e3
        cool_tot_demand = P.weighted_sum(c_demands, symbolic=False) / 1e3

        if typ_days == 'None':

            ### Operational costs - Total commodities - no typical days
            regex_price = 'el_ind_price|el_cons_price|gas_price'
            regex_commodities = 'el_ind_use|el_cons_use|gas_use'
            commodities = params.filter(regex = regex_price).join(operation.filter(regex = regex_commodities))
            total_commodities = (commodities.filter(items=['Source_el_cons_price','Source_el_cons_use','weights']).prod(axis=1)+commodities.filter(items=['Source_el_ind_price','Source_el_ind_use','weights']).prod(axis=1) + commodities.filter(like='gas').prod(axis=1)).sum()

            ## Operational costs - Op costs per components
            regex_op_costs = 'Inflow_pipe_P|Return_pipe_P|ASHP_C_cons_P_HP|HP_LC_C_cons_P_HP|HS_el_C_cons_q_dot|Source_gas_use' ### using gas use instead of el_gas_Qdot because of boiler efficiency
            op_costs = params.filter(regex = regex_price).join(operation.filter(regex = regex_op_costs))
            op_costs_comp = pd.DataFrame(columns=comp_list)
        else:
            scenarios = P.scenario_weights
            scenarios_dict = scenarios.to_frame().to_dict()['pi']
            ### Operational costs - Total commodities
            regex_price = 'el_ind_price|el_cons_price|gas_price'
            regex_commodities = 'el_ind_use|el_cons_use|gas_use'
            commodities = params.filter(regex = regex_price).join(operation.filter(regex = regex_commodities))
            commodities.reset_index(inplace=True)
            commodities['weights'] = commodities['s'].map(scenarios_dict.get).apply(lambda x: x)
            total_commodities = (commodities.filter(items=['Source_el_cons_price','Source_el_cons_use','weights']).prod(axis=1)+commodities.filter(items=['Source_el_ind_price','Source_el_ind_use','weights']).prod(axis=1) + commodities.filter(like='gas').prod(axis=1)).sum()

            ## Operational costs - Op costs per components
            regex_op_costs = 'Inflow_pipe_P|Return_pipe_P|ASHP_C_cons_P_HP|HP_LC_C_cons_P_HP|HS_el_C_cons_q_dot|Source_gas_use' ### using gas use instead of el_gas_Qdot because of boiler efficiency
            op_costs = params.filter(regex = regex_price).join(operation.filter(regex = regex_op_costs))
            op_costs.reset_index(inplace=True)
            op_costs['weights'] = commodities['s'].map(scenarios_dict.get).apply(lambda x: x)
            op_costs_comp = pd.DataFrame(columns=comp_list)

        cons_n = params.filter(regex = '_cons_n').iloc[0,0]
        boiler_design = design.filter(regex = 'HS_gas_C_cons_Qdot_max').iloc[0,0]
        ashp_design = design.filter(regex = 'ASHP_C_cons_Qdot_design').iloc[0,0]
        print(f" Boiler = {boiler_design}, ASHP = {ashp_design}, el_price = {commodities.filter(regex = 'el_cons_price').iloc[0,0]}")
        Dic_costs[pset_id] = {
        'tot_dem': tot_demand,
        'h_dem':heat_tot_demand,
        'c_dem':cool_tot_demand,
        'boiler_des': boiler_design,
        'ashp_des':ashp_design}
    
    df_costs = pd.DataFrame.from_dict(Dic_costs).T.reset_index(drop = True)

    pset_post = pd.concat([
                        pset.reset_index(drop=True), # original pset
                        df_costs.reset_index(drop = True), ## Total Costs
                        ], axis = 1)



    pset_post = pset.reset_index(drop=True), # modified pset
    print(pset_post)
    print(pset_post['boiler_des'])
    
    pickledir_post = parent_path / '_results' / str(run + '_post_processed' + '.pk')
    with pickledir_post.open('wb') as fp:
        pickle.dump((pset_post), fp)
    return pset_post


if __name__ == '__main__' and reprocessing == False:
    import psweep as ps
    from _scripts.model import instantiate_model, run_model
    from pathlib import Path
    import pandas as pd
    import os
    import pickle
    import numpy as np

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


    data_path = parent_path / '_data'
    data_path_VIC = data_path / 'VIC'
    data_path_NRW = data_path / 'NRW'

    # Region Checker
    if region == 'VIC':
        elec_cons = elec_cons_VIC
        elec_ind = elec_ind_VIC
        elec_fac = elec_fac_VIC
        gas_cons = elec_cons_VIC
        gas_ind = elec_ind_VIC
        gas_fac = elec_fac_VIC
        elec_gas_ratio = elec_cons / gas_cons

    elif region =='NRW':
        elec_cons = elec_cons_NRW
        elec_ind = elec_ind_NRW
        elec_fac = elec_fac_NRW
        gas_cons = elec_cons_NRW
        gas_ind = elec_ind_NRW
        gas_fac = elec_fac_NRW
        elec_gas_ratio = elec_cons / gas_cons

    #instantiate energy Systems
    n_long = [1]
    el_price_list = np.linspace(0.1,0.5,41).tolist()
    print(el_price_list)
    el_price_list.append(elec_cons/100)
    gas_price_list = [gas_cons/100]
    el_price_param = ps.plist('el_price', el_price_list)
    gas_price_param = ps.plist('gas_price', gas_price_list)
    n_con_param = ps.plist('n_cons',n_long)
    typ_days = ps.plist('typical_days',[typ_day_file]) # Either 'None' for a complete timeseries (8760 hrs) or the filename with the aggregation
    ES = instantiate_model() #either reversible, only heat or only cool. if not specified the ashp is reversible.
    

    params = ps.pgrid(el_price_param,n_con_param,typ_days)
    df = ps.run_local(calc_func, 
                    params = params,
                    database_dir = str(parent_path / '_results'),
                    calc_dir= str(parent_path / '_results' / run),
                    database_basename= run + '.pk',
                    simulate = False,
                    save = True,
                    )
    print(df)
    df_post = post_process_cost(df,run)
    print(df_post)



if __name__ == '__main__' and reprocessing == True:
    import psweep as ps
    from pathlib import Path
    import pandas as pd
    import os  
    import pickle

    print('Reprocessing ' + str(' '.join(repro_name_list)) + ' Results. Cancel if not it is not the intended script execution.')
    #reprocessing already ran sets
    results_path = parent_path / '_results'

    os.chdir(str(parent_path))

    ##### list of files
    #NRW
    if not repro_name_list:
        pass
    else:
        for name in repro_name_list:
            dirfile = [f for f in results_path.glob('*' + name +'.pk')]
            print(dirfile)
            with open(f'{dirfile[0]}', 'rb') as f: # last generated pickle
                db = pickle.load(f)
            db_post = post_process_cost(db,name)
            print(db_post)