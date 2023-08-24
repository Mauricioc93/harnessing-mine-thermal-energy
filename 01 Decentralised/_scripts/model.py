"""Model generator for DECEN case of analysis"""
# Author: Mauricio Carcamo
import sys
from pathlib import Path

if __name__ == '__main__':
    parent_path = Path(__file__).parents[1]
else:
    parent_path = Path(__file__).parents[1]

data_path = parent_path / '_data'

sys.path.insert(0,str(Path(__file__).parents[0])) ## adding path to _script subfolder to importing other modules
sys.path.insert(0,str(Path(__file__).parents[1])) ## adding path to _script subfolder to importing other modules

from math import nan
import pandas as pd
import numpy as np
import gurobipy as gp
import os
from comando.core import System
from comando.interfaces.gurobi import to_gurobi
from comando.interfaces.pyomo import to_pyomo
from pyomo.environ import value
from comando.utility import make_tac_objective
from utilities import slope_intercept,segments_fit
# from comando.utility import make_tac_objective_lifetime

SEED = 123

def pipe_parameters(params,design_pipe,pipe_name,n_seg,L, ins = 'NoIns'):

    pipe_cost = design_pipe['Cost_'+ ins].values[0] # cost per metre [$/m]
    m_max = design_pipe['Max_mflow'].values[0] # maximum mass flow [kg/s]
    v_max = design_pipe['vmax'].values[0] # maximum velocity [m/s]
    D_i = design_pipe['Di'].values[0] # pipe inner diameter [m]
    fric = design_pipe['Friction_factor'].values[0] # friction factor
    # U = design_pipe['U_' + ins].values[0] # U_value [W/m/K]
    R = design_pipe['Res_' + ins].values[0]
    thi = design_pipe['s'].values[0] # thick_value [m]
    eta_pump = 0.72
    rho = 998
    cp = 4.184

    # pipe parameters
    params[pipe_name+'_pipe_cost'] = pipe_cost
    params[pipe_name+'_m_max'] = m_max
    params[pipe_name+'_v_max'] = v_max
    params[pipe_name+'_D_i'] = D_i
    params[pipe_name+'_friction_factor'] = fric
    # params[pipe_name+'_U'] = U
    params[pipe_name+'_thi'] = thi
    params[pipe_name+'_eta_pump'] = eta_pump

    # pumping
    a_v,b_v = seg_power(fric,D_i,rho,eta_pump,m_max,n_seg)
    for i,(a,b) in enumerate(zip(a_v,b_v)):
        params[pipe_name+'_a'+str(i)] = a
        params[pipe_name+'_b'+str(i)] = b

    # heat losses
    a_hv,b_hv,p_m = seg_massflow(R, rho, cp, m_max, L)
    if p_m < 0.1*m_max:
        p_m = 0.1*m_max
    params[pipe_name + '_p_m'] = p_m
    params[pipe_name+'_a_hv'] = a_hv[1] #using last section only
    params[pipe_name+'_b_hv'] = b_hv[1] #using last section only
    return params

def seg_power(fric,D_i,rho,eta_pump,m_max,n_seg):
    n_vect_x = 100 # discretisation (x) for function evaulation f(x)
    power_L_func = lambda m: (8*fric/(D_i**5*rho**2*np.pi**2)*(m)**3)/eta_pump # cubic function P_el = f(mdot) kW
    m_vect = np.linspace(0,m_max, n_vect_x) #mdot discretisation
    power_vect = power_L_func(m_vect) #elec power evaluation
    px, py = segments_fit(m_vect, power_vect, n_seg) ## stepwise fitting
    #force origin through zero
    px[0] = 0
    py[0] = 0
    for x,y in zip(px,py):
        if y < power_L_func(x):
            y = power_L_func(x)
    a_list = []
    b_list = []
    for i, (x,y) in enumerate(zip(px,py)): # collection of slope and intercept values for linearization.
        if i == 0:
            continue
        else:
            a,b = slope_intercept(px[i-1],py[i-1],x,y) 
            a_list.append(a)
            b_list.append(b)
    return a_list, b_list

def seg_massflow(R,rho,cp,m_max,L):
    nsec = 2 # fixed to 2
    ## mass flow vector segmentation
    dm = 0.05
    n = int(np.round(m_max/dm,0))
    m_list = np.linspace(dm,m_max,n)
    Y_L_sing = np.fromiter((np.exp(-L/(rho*cp*m*R)) if m>0 else 0 for m in m_list),float)
    px, py = segments_fit(m_list, Y_L_sing, nsec)
    py[0] = Y_L_sing[0] # fix to m = 0
    py[2] = Y_L_sing[-1] # fix to m = m_ma
    p_m = px[1] ##breakpoint
    a_list = []
    b_list = []

    for i, (x,y) in enumerate(zip(px,py)):
        if i == 0:
            continue
        elif i == 1:
            a,b = slope_intercept(x,y,px[i-1],py[i-1])
            a_list.append(a)
            b_list.append(Y_L_sing[0])
        else:
            a,b = slope_intercept(x,y,px[i-1],py[i-1])
            a_list.append(a)
            b_list.append(b)
    return a_list,b_list,p_m





def C2K(temperature):
    """Convert temperature from Celsius to Kelvin."""
    return temperature + 273.15

def create_energysystem(consumers,case=None, generic_comp = True, **kwargs):
    """Create the an energy system model based on demand data.

    Arguments
    ---------
    consumers: Iterable
        Contains the names of consumers to be considered
    """
    import importlib
    # import components
    if generic_comp == True:
        comp_module = importlib.import_module(f'_scripts.components',package=False) ## case folder
    else:
        if case == None:
            ValueError('Case name not assigned!')
        else:
            comp_module = importlib.import_module(f'_scripts.{case}.components',package=False) ## case folder

    DummySource = comp_module.DummySource

    DummySink = comp_module.DummySink
    FreeMassflowSource = comp_module.FreeMassflowSource
    NetworkPipe = comp_module.NetworkPipe
    Consumer = comp_module.Consumer
    ashp_mode = kwargs.get('ashp_mode','reversible')
    del comp_module ### not needed after importing components
    label = kwargs.get('label','ES')
    # Instatiate energy system
    ES = System(label=label)
    # Component names

    names = {# Sources
            'Source_el_cons' : label + 'Source_el_cons',
            'Source_el_ind': label + 'Source_el_ind',
            'Source_gas': label + 'Source_gas',
            'Source_wh': label + 'Source_wh',
            # Sinks
            'Sink_wh': label + 'Sink_wh',
            # Pipes
            'Inflow_pipe': label + 'Inflow_pipe',
            'Return_pipe': label + 'Return_pipe',
            #Rest of transformers are added with the consumer
        }
    # Instantiate central components 
    source_el_cons = DummySource(names['Source_el_cons'])
    source_el_ind = DummySource(names['Source_el_ind'])
    source_gas = DummySource(names['Source_gas'])

    # Instantiate source options TODO add extra source with for pumping
    source_wh = FreeMassflowSource(names['Source_wh'])
    sink_wh = DummySink(names['Sink_wh'])

    # Instantiate transmission pipes - Load pipe data
    n_seg = kwargs.get('n_seg',10)
    inflow_pipe = NetworkPipe(names['Inflow_pipe'], type = 'pipe_losses', n_seg = n_seg)
    return_pipe = NetworkPipe(names['Return_pipe'], type = 'pipe_losses', n_seg = n_seg)


    # Add central components and network to comp
    comp = [source_el_cons,
            source_el_ind,
            source_gas,
            source_wh,
            sink_wh,
            inflow_pipe,
            return_pipe]


    # Set connections of central components
    conn = {
         # HP - Circulation - Power
        'el_ind_Bus': [source_el_ind.OUT,
                   inflow_pipe.IN_P_pump,
                   return_pipe.IN_P_pump],
        'el_cons_Bus': [source_el_cons.OUT],
        'gas_Bus': [source_gas.OUT],
        #  Supply direction - Mine to cons
        'mw_sup_in_Bus': [source_wh.OUT_mflow,
                      inflow_pipe.IN_mflow],
        'mw_sup_out_Bus': [inflow_pipe.OUT_mflow],
            #  Return direction - Cons to mine
        'mw_ret_in_Bus': [return_pipe.IN_mflow],
        'mw_ret_out_Bus': [return_pipe.OUT_mflow,
                            sink_wh.IN],

        # Temperature Pipe-Cons     
        # 'pipe_sup_cons_t_Bus': [inflow_pipe.OUT_t],
        # 'pipe_ret_cons_t_Bus': [return_pipe.IN_t],
    }
    dt_max = kwargs.get('dt_max', 25)
    ES.add_eq_constraint(source_wh['tflow'],inflow_pipe['t_in'], 'flow_temp_eq_in_flow_pipe')
    # Add consumers and set connections of decentral components
    for consumer in consumers:
        cons = Consumer(f'C_{consumer}', prefix = label, ashp_mode = ashp_mode, dt_max = dt_max)
        comp.append(cons)

        cons.extend_connection('IN_P_el') # electricity input

        #connection to busses
        conn['el_cons_Bus'].append(cons.IN_P_el)
        # conn['gas_Bus'].append(cons.IN_P_gas)
        conn['mw_sup_out_Bus'].append(cons.IN_mflow_a_h)
        conn['mw_sup_out_Bus'].append(cons.IN_mflow_a_c)
        conn['mw_ret_in_Bus'].append(cons.OUT_mflow_a_h)
        conn['mw_ret_in_Bus'].append(cons.OUT_mflow_a_c)


        for subcomp in cons.components:
            if 'HX_LC' in  subcomp.label:
                HX_cons_t_in_a = subcomp.__getitem__('t_in_a')
                HX_cons_t_out_a = subcomp.__getitem__('t_out_a')
                HX_op = subcomp.__getitem__('b_op')
                HX_b = subcomp.__getitem__('b_build')

            if 'WSHP_LC' in subcomp.label:
                HP_cons_t_in_a = subcomp.__getitem__('t_in_a')
                HP_cons_t_out_a_h = subcomp.__getitem__('t_out_a_h')
                HP_cons_t_out_a_c = subcomp.__getitem__('t_out_a_c')
                HP_op_h = subcomp.__getitem__('b_h')
                HP_op_c = subcomp.__getitem__('b_c')
                HX_b = subcomp.__getitem__('b_build')
            names[subcomp.label.replace(label, '')] = subcomp.label
                

        ES.add_eq_constraint(inflow_pipe['t_out'],(HX_cons_t_in_a-273.15)*HX_op+((HP_cons_t_in_a-273.15)*HP_op_h+(HP_cons_t_in_a-273.15)*HP_op_c)+273.15, 'inflow_temp_eq_out_flow_pipe')
        ES.add_eq_constraint(return_pipe['t_in'],(HX_cons_t_out_a-273.15)*HX_op+((HP_cons_t_out_a_h-273.15)*HP_op_h+(HP_cons_t_out_a_c-273.15)*(1-HP_op_h))+273.15, 'outflow_temp_eq_out_flow_pipe')

    for c in comp:
        ES.add(c)
    for bus_id, connectors in conn.items():
        ES.connect(bus_id, connectors)      

    return ES,{label: names}

def instantiate_model(**kwargs):
    """instantiate the Decen Model"""
    # Create energy system
    consumer_groups = ['cons']
    ES,names = create_energysystem(consumers=consumer_groups, **kwargs)
        # Add expressions to ES
    for expr_id in ['investment_costs', 'fixed_costs', 'variable_costs']:
        ES.add_expression(expr_id, ES.aggregate_component_expressions(expr_id))
    return ES,names
    

def run_model(ES, operational_calc=False, case = None, typ_days=None, interface = 'gurobi', **kwargs):
    """Run the model

    """
    import warnings 
    from pathlib import Path
    import os
    import gurobipy as gp
    # location of parent path
    parent_path = Path(__file__).parents[1]
    data_path = parent_path / '_data'
    cost_data_path = data_path

    # Case dependant fixed variables - thermal loads, ambient temperature
    if case != None:
        case_data_path = data_path / case # case data folder

    else:
        ValueError('case not assigned!')

    # Demand data obtain from bin method for SFH

    if typ_days is None:
        data = pd.read_csv(case_data_path / 'time_series_inp.csv', index_col = 0)
    else:
        data = pd.read_csv(case_data_path / typ_days, sep = ',', index_col = 0)
        data.set_index(['s', 'TimeStep'], inplace=True)
        scenarios = data['period_weight'].groupby('s').first().rename('pi')
        # print(scenarios)
    ambient_T = data['Dry_Bulb_Temperature']
    T_g = data['tg']
    timesteps = data['dt']


    # Collect parameters of all components
    params = dict()


    ### Price variables - commodities
    price_data = pd.read_csv(cost_data_path / 'price_inp.csv', index_col = 0) # table of components base costs
    
    #Component           MAINTENANCE COSTS | NOMINAL COSTS | VARIABLE COSTS
    ashp_costs = kwargs.get('ashp_costs',price_data.loc['ashp','maintenance_cost':'variable_cost'])
    wshp_costs = kwargs.get('decen_wshp_costs',price_data.loc['decen_wshp','maintenance_cost':'variable_cost'])
    cen_wshp_costs = kwargs.get('cen_wshp_costs',price_data.loc['cen_wshp','maintenance_cost':'variable_cost'])
    boiler_costs = kwargs.get('boiler_costs',price_data.loc['boiler','maintenance_cost':'variable_cost'])
    el_heat_costs = kwargs.get('el_heat_costs',price_data.loc['el_heat','maintenance_cost':'variable_cost'])
    heat_ex_costs = kwargs.get('heat_ex_costs',price_data.loc['heat_ex','maintenance_cost':'variable_cost'])

    ## case variables - specific to the location VIC or NRW

    el_price = kwargs.get('el_price',31.12/100) # [€/kWh]
    elec_factor = kwargs.get('elec_factor',1)
    gas_price = kwargs.get('gas_price',10.59/100)  # [€/kWh]
    n_con = kwargs.get('n_cons',1000) # number of consumers
    wh_temp = kwargs.get('T_source',10+273.15) 
    wh_height = kwargs.get('h_pump',0)

    #comp operation variables
        # dt Heatpumps
    # hp_a_dt_h = dt # heat extraction - heating
    # hp_a_dt_c = -dt # heat rejection - cooling



    # efficiencies
    ashp_efficiency = 0.1 # turned off
    boiler_eff = 0.1
    el_heat_eff = 0.1

    # wshp
    wshp_efficiency = 0.50

    #Sources parameters
    
    params['Source_el_cons_price'] = el_price
    params['Source_el_ind_price'] = el_price*elec_factor
    params['Source_gas_price'] = gas_price
    params['Source_wh_T_price'] = wh_height*9.81*el_price*elec_factor 
    params['Source_wh_Tset'] = wh_temp
    
    #Network costs
    length_pipe = kwargs.get('length',100)
    DN = kwargs.get('DN',400)
    ins = kwargs.get('ins', 'NoIns') # options, NoIns, S1, S2, S3. NoIns = PE pipe, SX = Steel pipe with insulation. Default DECEN: None, Default CEN: S2
    if ins == 'NoIns':
        pipe_data = pd.read_csv(cost_data_path / 'pipe_PE_prop.csv', sep=',') # No insulation PE pipes
    elif ins == 'S1' or ins == 'S2' or ins == 'S3':
        pipe_data = pd.read_csv(cost_data_path / 'pipe_steel_prop.csv', sep=',') # Steel pipes with different insulation thickness
    else:
        ValueError('options for insulation are: NoIns, S1, S2, S3. incorrect Input')
    design_pipe = pipe_data[pipe_data['DN'] == DN] ## FIXED for the moment

    #Network Pump & Heat Losses/Gains parameters
    n_seg = kwargs.get('n_seg',10) # for pumping parameters
    params = pipe_parameters(params,design_pipe,'Return_pipe',n_seg,length_pipe, ins)
    params = pipe_parameters(params,design_pipe,'Inflow_pipe',n_seg,length_pipe, ins)


    params['Inflow_pipe_length'] = length_pipe
    params['Return_pipe_length'] = length_pipe
    params['Inflow_pipe_T_g'] = C2K(T_g)
    params['Return_pipe_T_g'] = C2K(T_g)

    ## No centralised Heat Pump

   # Consumer settings
    consumer_groups = ['cons'] #single consumer group
    for consumer_group in consumer_groups:
        # # HeatPump/HX Settings


        # params[f'LC_C_{consumer_group}_dT_a_c'] = hp_a_dt_h
        params[f'LC_C_{consumer_group}_n'] = n_con
        params[f'LC_C_{consumer_group}_cp'] = 4.184

        # params[f'WSHP_LC_C_{consumer_group}_dT_a_h'] = hp_a_dt_h
        # params[f'WSHP_LC_C_{consumer_group}_dT_a_c'] = hp_a_dt_c
        params[f'WSHP_LC_C_{consumer_group}_n'] = n_con
        params[f'WSHP_LC_C_{consumer_group}_p_spec'] = wshp_costs['variable_cost']
        params[f'WSHP_LC_C_{consumer_group}_p_fix'] = wshp_costs['nominal_cost'] # maintenance costs are a 2.5% of the total investment cost, check components.py
        params[f'WSHP_LC_C_{consumer_group}_p_main'] = wshp_costs['maintenance_cost']
        params[f'WSHP_LC_C_{consumer_group}_eta_h'] = wshp_efficiency
        params[f'WSHP_LC_C_{consumer_group}_eta_c'] = wshp_efficiency
        params[f'WSHP_LC_C_{consumer_group}_cp'] = 4.184

        params[f'HX_LC_C_{consumer_group}_dT_a_c'] = 5.0
        params[f'HX_LC_C_{consumer_group}_n'] = n_con
        params[f'HX_LC_C_{consumer_group}_cp'] = 4.184

        params[f'ASHP_C_{consumer_group}_dT_a_h'] = 10.0
        params[f'ASHP_C_{consumer_group}_dT_a_c'] = -10.0
        params[f'ASHP_C_{consumer_group}_n'] = n_con
        params[f'ASHP_C_{consumer_group}_p_spec'] = ashp_costs['variable_cost']
        params[f'ASHP_C_{consumer_group}_p_fix'] = ashp_costs['nominal_cost']
        params[f'ASHP_C_{consumer_group}_p_main'] = ashp_costs['maintenance_cost']
        params[f'ASHP_C_{consumer_group}_T_amb'] = C2K(ambient_T)
        params[f'ASHP_C_{consumer_group}_eta_h'] = ashp_efficiency
        params[f'ASHP_C_{consumer_group}_eta_c'] = ashp_efficiency
        params[f'ASHP_C_{consumer_group}_cp'] = 4.184

        # BES settings
        #heat load
        params[f'BES_C_{consumer_group}_heating_Qdot'] = \
            data[f'heat_load'] *n_con 
        params[f'BES_C_{consumer_group}_heating_T_flow'] = C2K(55)
        #cool load
        params[f'BES_C_{consumer_group}_cooling_Qdot'] = \
            data[f'cool_load'] *n_con*-1
        params[f'BES_C_{consumer_group}_cooling_T_flow'] = C2K(7)
        # Resistance heating
        params[f'HS_el_C_{consumer_group}_n'] = n_con
        params[f'HS_el_C_{consumer_group}_p_spec'] = el_heat_costs['variable_cost']
        params[f'HS_el_C_{consumer_group}_p_fix'] = el_heat_costs['nominal_cost']
        params[f'HS_el_C_{consumer_group}_p_main'] = el_heat_costs['nominal_cost']
        params[f'HS_el_C_{consumer_group}_efficiency'] = el_heat_eff # Resistance heating
        # Boiler
        params[f'HS_gas_C_{consumer_group}_n'] = n_con
        params[f'HS_gas_C_{consumer_group}_p_spec'] = boiler_costs['variable_cost']
        params[f'HS_gas_C_{consumer_group}_p_fix'] = boiler_costs['nominal_cost']
        params[f'HS_gas_C_{consumer_group}_p_main'] = boiler_costs['maintenance_cost']
        params[f'HS_gas_C_{consumer_group}_efficiency'] = boiler_eff # Gas boiler


    # update params keys to assign to model

    run_label = kwargs.get('run_label','')
    params_upd = {(run_label + k) : v  for (k,v) in params.items() }

    # create Problem

    if typ_days is None:
        ##This uses whole year \ disconnected days
        P = ES.create_problem(
            *make_tac_objective(ES, n=30, i=0.03),
            timesteps=timesteps,
            data=params_upd,
            name='heat_and_cool_whole_year'
        )
    else:
        # This uses typical days and scenario
        P = ES.create_problem(
            *make_tac_objective(ES, n=30, i=0.03),
            timesteps=timesteps,
            scenarios=scenarios,
            data=params_upd,
            name='heat_and_cool_typical_day'
        )

    # if operational_calc:
    #     # set design variables according to design.csv
    #     result_df = pd.read_csv('design_inp.csv', index_col='name')
    #     for dv in P.design_variables:
    #         dv.value = result_df.loc[dv.name]['value']
    #         dv.fix()
    #         print(f'set {dv.name} to {dv.value}')

    ###########
    ## 1 Module
    ###########


    # if interface == 'gurobi':
    #     m = to_gurobi(P) # straight to Gurobi - not possible for time discr.
    #     options = dict(  # Options assuming Gurobi 9.5.1
    #             Seed=SEED,
    #             NonConvex=2,
    #             MIPGap=0.01,
    #             MIPFocus=1,
    #             timelimit=10800,
    #             # LogFile=log_name,
    #             OutputFlag=0,
    #         )
    #     # m.computeIIS() # in case problem is infeasible
    #     # m.write('m.ilp')
    #     m.solve_multi(**options)
    # elif interface == 'pyomo':
    #     m = to_pyomo(P)
    #     options = dict(  # Options assuming Gurobi 9.1.1
    #         Seed=SEED,
    #         NonConvex=2,
    #         MIPGap=0.01,
    #         MIPFocus=1,
    #         timelimit= 10800,
    #         # LogFile=log_name,
    #         OutputFlag=0,
    #     )

    
    #     m.solve(solver='gurobi',options=options)

    # return P, m

    ##########
    # Multiprocessing
    ##########

    return P

if __name__ == '__main__':
    import argparse
    import pathlib
    import os
    import pandas as pd
    from pyomo.environ import value
    from _scripts.utilities import annuity

    def find_neighbours(value, df, colname):
        exactmatch = df[df[colname] == value]
        if not exactmatch.empty:
            return exactmatch.index
        else:
            upperneighbour_ind = df[df[colname] > value][colname].idxmin()
            return upperneighbour_ind

    def DN_calc(dT,T,q_c,q_h,n_con,pipeset, cooling = 'yes'):
        #heating
        cp = 4.184
        eff = 0.50
        Th_cold = T
        Th_hot = 55+273.15
        COP_H =Th_hot/(Th_hot-Th_cold)*eff
        m_max_h = q_h/(dT*cp)*(1-1/COP_H)*1.*n_con

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

    if __package__ is None:
        import sys
        from pathlib import Path

        file = Path(__file__).resolve()
        sys.path.append(str(file.parents[1]))
    import model
    __package__ = model.__name__
    ES = instantiate_model(ashp_mode = 'cool_only')
    length_in = 50
    q_c = 6.43
    q_h = 7.29
    n_con = 100
    dt = 8
    data_path = parent_path / '_data'
    pipe_data = pd.read_csv(data_path / 'pipe_prop.csv', sep=',')
    DN_in = DN_calc(10,5+273.15,q_c,q_h,n_con,pipe_data, cooling = 'yes')
    el_price_in = 30/100

    P,m = run_model(ES,case = 'VIC',
                                    typ_days = None, 
                                    interface = 'gurobi',
                                     length = length_in, DN = DN_in, n_cons = n_con, el_price = el_price_in)


    # h_demands = sum(p for p in P.parameters if p.name.endswith('heating_Qdot'))
    # c_demands = sum(p for p in P.parameters if p.name.endswith('cooling_Qdot'))
    # tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3
   ### Costs
    params = P.data.getter()
    operation = P.operation.transpose()
    design = P.design.transpose()

    # tac = value(m.objective)
    tac = m.ObjVal
    status = m.Status
    print(status)
    print(design)