"""Model generator for SFH case of analysis"""
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
import os
from comando.core import System
from comando.interfaces.gurobi import to_gurobi
from comando.interfaces.pyomo import to_pyomo
from pyomo.environ import value
from comando.utility import make_tac_objective
from utilities import slope_intercept,segments_fit
# from comando.utility import make_tac_objective_lifetime






SEED = 123


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
    Consumer = comp_module.Consumer
    ashp_mode = kwargs.get('ashp_mode','reversible')
    del comp_module ### not needed after importing components

    # Instatiate energy system
    ES = System(label='ES')
    # Instantiate central components 
    source_el_cons = DummySource('Source_el_cons')
    source_el_ind = DummySource('Source_el_ind')
    source_gas = DummySource('Source_gas')

    # Add central components and network to comp
    comp = [source_el_ind,
            source_el_cons,
            source_gas]


    # Set connections of central components
    conn = {
         # HP - Circulation - Power
        'el_cons_Bus': [source_el_cons.OUT],
        'gas_Bus': [source_gas.OUT],
        }

    # Add consumers and set connections of decentral components
    for consumer in consumers:
        cons = Consumer(f'C_{consumer}',ashp_mode = ashp_mode)
        comp.append(cons)

        cons.extend_connection('IN_P_el') # electricity input

        #connection to busses
        conn['el_cons_Bus'].append(cons.IN_P_el)
        conn['gas_Bus'].append(cons.IN_P_gas)




    for c in comp:
        ES.add(c)
    for bus_id, connectors in conn.items():
        ES.connect(bus_id, connectors)      

    return ES

def instantiate_model(**kwargs):
    """instantiate the Decen Model"""
    # Create energy system
    consumer_groups = ['cons']
    ES = create_energysystem(consumers=consumer_groups, **kwargs)
        # Add expressions to ES
    for expr_id in ['investment_costs', 'fixed_costs', 'variable_costs']:
        ES.add_expression(expr_id, ES.aggregate_component_expressions(expr_id))
    return ES
    

def run_model(ES, operational_calc=False, case = None, typ_days=None, interface = 'gurobi', **kwargs):
    """Run the SFH case.

    """
    import warnings 
    from pathlib import Path
    import os

    # location of parent path
    parent_path = Path(__file__).parents[1]
    


    data_path = parent_path / '_data'


    # Case dependant fixed variables - loads, ambient temperature
    if case != None:
        data_path = data_path / case # case data folder
    else:
        ValueError('case not assigned!')


    # Demand data obtain from bin method for SFH

    if typ_days is None:
        data = pd.read_csv(data_path / 'time_series_inp.csv', index_col = 0)
    else:
        data = pd.read_csv(data_path / typ_days, sep = ',', index_col = 0)
        data.set_index(['s', 'TimeStep'], inplace=True)
        scenarios = data['period_weight'].groupby('s').first().rename('pi')
    ambient_T = data['Dry_Bulb_Temperature']
    timesteps = data['dt']


    # Collect parameters of all components
    params = dict()


    ### Price variables - commodities
    price_data = pd.read_csv(data_path / 'price_inp.csv', index_col = 0) # table of components base costs
    
    #Component          | FIX COSTS | VARIABLE COSTS | MAINTENANCE COSTS (TBA)

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
    wh_temp = kwargs.get('T_source',10+273.15) ############################ CORREGIR

    
    #comp operation variables
        # dt Heatpumps
    hp_a_dt_h = 10.0 # heat extraction - heating
    hp_a_dt_c = -10.0 # heat rejection - cooling

    # efficiencies
    ashp_efficiency = 0.4
    boiler_eff = 0.97
    el_heat_eff = 1.0

    # wshp
    eta_h = 0.50
    eta_c = 0.50

    #Sources parameters
    
    params['Source_el_cons_price'] = el_price
    params['Source_el_ind_price'] = el_price*elec_factor
    params['Source_gas_price'] = gas_price
    params['Source_wh_T_price'] = 0
    params['Source_wh_Tset'] = wh_temp 
    


   # Consumer settings
    consumer_groups = ['cons'] #single consumer for the moment
    for consumer_group in consumer_groups:
        # # HeatPump/HX Settings


        params[f'LC_C_{consumer_group}_dT_a_c'] = 10
        params[f'LC_C_{consumer_group}_n'] = n_con
        params[f'LC_C_{consumer_group}_cp'] = 4.184

        params[f'ASHP_C_{consumer_group}_dT_a_h'] = hp_a_dt_h
        params[f'ASHP_C_{consumer_group}_dT_a_c'] = hp_a_dt_c
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
        params[f'HS_el_C_{consumer_group}_p_spec'] = el_heat_costs['variable_cost'] + 0.1
        params[f'HS_el_C_{consumer_group}_p_fix'] = el_heat_costs['nominal_cost']
        params[f'HS_el_C_{consumer_group}_p_main'] = el_heat_costs['nominal_cost']
        params[f'HS_el_C_{consumer_group}_efficiency'] = el_heat_eff # Resistance heating
        # Boiler
        params[f'HS_gas_C_{consumer_group}_n'] = n_con
        params[f'HS_gas_C_{consumer_group}_p_spec'] = boiler_costs['variable_cost'] + 0.1
        params[f'HS_gas_C_{consumer_group}_p_fix'] = boiler_costs['nominal_cost']
        params[f'HS_gas_C_{consumer_group}_p_main'] = boiler_costs['maintenance_cost']
        params[f'HS_gas_C_{consumer_group}_efficiency'] = boiler_eff # Gas boiler

    

    # create Problem

    if typ_days is None:
        ##This uses whole year \ disconnected days
        P = ES.create_problem(
            *make_tac_objective(ES, n=30, i=0.03),
            timesteps=timesteps,
            data=params,
            name='heat_and_cool_whole_year'
        )
    else:
        # This uses typical days and scenario
        P = ES.create_problem(
            *make_tac_objective(ES, n=30, i=0.03),
            timesteps=timesteps,
            scenarios=scenarios,
            data=params,
            name='heat_and_cool_whole_year'
        )

    
    if interface == 'gurobi':
        m = to_gurobi(P) # straight to Gurobi - not possible for time discr.
        options = dict(  # Options assuming Gurobi 9.5.1
                Seed=SEED,
                NonConvex=2,
                MIPGap=0.01,
                MIPFocus=1,
                timelimit=10800,
                # LogFile=log_name,
                OutputFlag=1,
            )
        # m.computeIIS() # in case problem is infeasible
        # m.write('m.ilp')
        m.solve(**options)
    elif interface == 'pyomo':
        m = to_pyomo(P)
        options = dict(  # Options assuming Gurobi 9.1.1
            Seed=SEED,
            NonConvex=2,
            MIPGap=0.01,
            MIPFocus=1,
            timelimit= 10800,
            # LogFile=log_name,
            OutputFlag=1,
        )
        m.solve(solver='gurobi',options=options)


    return P, m

if __name__ == '__main__':
    import argparse
    import pathlib
    import os
    import pandas as pd
    from pyomo.environ import value
    from _scripts.utilities import annuity

    if __package__ is None:
        import sys
        from pathlib import Path

        file = Path(__file__).resolve()
        sys.path.append(str(file.parents[1]))
    import model

    __package__ = model.__name__
    ES = instantiate_model()
    # P, m = run_model(ES,operational_calc=False, case = 'NRW', typ_days='Medoid_12d_norescale.csv', interface = 'pyomo',**pset) ## Remember to instantiate model
    # def run_model(ES, operational_calc=False, case = None, typ_days=None, interface = 'gurobi', **kwargs):
    length_in = 1
    DN_in = 32
    el_price_in = 30/100

    P,m = run_model(ES,case = 'NRW',
                                    typ_days = 'Medoid_4d_norescale.csv', 
                                    interface = 'pyomo',
                                     length = length_in, DN = DN_in, n_cons = 1, el_price = el_price_in)
    P.design.transpose().to_csv('design_check_typ.csv')
    P.operation.transpose().to_csv('operation_check_typ.csv')

    h_demands = sum(p for p in P.parameters if p.name.endswith('heating_Qdot'))
    c_demands = sum(p for p in P.parameters if p.name.endswith('cooling_Qdot'))
    tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3
   ### Costs
    params = P.data.getter()
    operation = P.operation.transpose()
    design = P.design.transpose()

    tac = value(m.objective)
    # tac = m.ObjVal


    ### Operational costs
    commodities = params.filter(regex = 'el_price|gas_price').join(operation.filter(regex = 'el_use|gas_use'))
    commodities.reset_index(inplace=True)
    op_costs = (commodities.filter(items=['Source_el_price','Source_el_use']).prod(axis=1) + commodities.filter(like='gas').prod(axis=1)).sum()

    ## Investment & Maintenance costs
    comp_list = ['HP', 'HS_gas','HS_el','Pipes']
    cons_n = params.filter(regex = '_cons_n').iloc[0,0]
    cost_inv = params.filter(regex = '_p_fix|_p_spec').iloc[0]

    cost_list = pd.DataFrame(index = ['Maintenance', 'Investment'], columns = comp_list)



    ### Total costs

    total_costs = {'Investment': annuity(cost_list.loc['Investment'].sum(axis=0),n=30,wacc = 0.03),'Maintenance': cost_list.loc['Maintenance'].sum(axis=0),'Operation':op_costs}
    print(total_costs)
    print(sum(total_costs.values()))
    tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3


    tac_2 = total_costs['Investment']+total_costs['Maintenance']+total_costs['Operation']
    print(f'\nAnnual energy demand: {tot_demand} MWh')
    print(f'\nTAC: {tac} AUD')
    print(f'\ncorresponds to: {tac / tot_demand} AUD/MWh')

