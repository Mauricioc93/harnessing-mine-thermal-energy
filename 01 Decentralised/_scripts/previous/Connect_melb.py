"""Case study for low temperature district heating network design."""
# This file is part of the COMANDO project which is released under the MIT
# license. See file LICENSE for full license details.
#
# AUTHORS: Marco Langiu, Dominik Hering
from math import nan
import pathlib
from tkinter.ttk import Separator
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial import QhullError
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from comando.core import System
from comando.interfaces.gurobi import to_gurobi
from comando.interfaces.pyomo import to_pyomo
from pyomo.environ import value
from comando.utility import make_tac_objective



data_path = pathlib.Path(__file__).parent

SEED = 123


def C2K(temperature):
    """Convert temperature from Celsius to Kelvin."""
    return temperature + 273.15

def create_energysystem(consumers):
    """Create the an energy system model based on demand data.

    Arguments
    ---------
    consumers: Iterable
        Contains the names of consumers to be considered
    """

     # Instatiate energy system
    ES = System(label="ES")


    # import components
    from TN_components import DummySource, Consumer, FreeMassflowSource, NetworkPipe
    # Instantiate central components
    source_el = DummySource('Source_el')
    source_gas = DummySource('Source_gas')
    source_wh = FreeMassflowSource('Source_wh')
    pipe_data = pd.read_csv(data_path / 'pipe_data.csv', sep=';')
    design_pipe = pipe_data[pipe_data['DN'] == 600]
    print(f"max mflow: {design_pipe['Max_mflow'].max()} kg/s")
    flow_pipe = NetworkPipe('Flow_pipe', pipe_data = design_pipe)
    
    # Add central components to comp
    comp = [source_el,
            source_gas,
            source_wh,
            flow_pipe]

    # Set connections of central components
    conn = {
                # HP - Power
        'el_Bus': [source_el.OUT,
                   flow_pipe.IN_P_pump],
        'gas_Bus': [source_gas.OUT],
        'mw_in_Bus': [source_wh.OUT_mflow,
                      flow_pipe.IN_mflow],
        'mw_out_Bus': [flow_pipe.OUT_mflow]
    }

    ES.add_eq_constraint(source_wh['tflow'],flow_pipe['t_in'], 'flow_temp_eq_in_flow_pipe')
    # Add consumers and set connections of decentral components
    for consumer in consumers:
        cons = Consumer(f'C_{consumer}')
        comp.append(cons)

        cons.extend_connection('IN_P_el') # electricity input


        conn['el_Bus'].append(cons.IN_P_el)
        conn['gas_Bus'].append(cons.IN_P_gas)
        conn['mw_out_Bus'].append(cons.IN_mflow_a_h)
        conn['mw_out_Bus'].append(cons.IN_mflow_a_c)
        
        for subcomp in cons.components:
            try:
                hp = subcomp.__getitem__('t_in_a')
            except:
                print('bipbop')
            else:
                hp = subcomp.__getitem__('t_in_a')
                print(hp)
                ES.add_eq_constraint(flow_pipe['t_out'],subcomp.__getitem__('t_in_a'), 'flow_temp_eq_out_flow_pipe')

        
        
    for c in comp:
        ES.add(c)
    for bus_id, connectors in conn.items():
        ES.connect(bus_id, connectors)        


        
    

    return ES


def run_destest_case_study(validate_result=False):
    """Run the single_house case.

    """
    from datetime import datetime
    import pickle
    import pathlib
    import os

    # Demand data obtain from bin method for SFH in Melb and Vict
    data = pd.read_csv(data_path / 'comando_inp_melb.csv', index_col=0)
    consumer_groups = ['heat_consumer']
    ES = create_energysystem(consumers=consumer_groups)
    print("\n\n\n=======================")
    print("Solving design problem.")
    print("=======================\n\n\n")

    # data = pd.read_csv(pathlib.Path(__file__).parent / 'Melb_Medoid_4d_24hrs.csv', sep=',')
    # data.set_index(['s', 'TimeStep'], inplace=True)
    # scenarios = data['period_weight'].groupby('s').first().rename('pi')

    ambient_T = data['Dry_Bulb_Temperature']
    timesteps = data['dt']

    # Add expressions to ES
    for expr_id in ['investment_costs', 'fixed_costs', 'variable_costs']:
        ES.add_expression(expr_id, ES.aggregate_component_expressions(expr_id))

    # Collect parameters of all components
    params = dict()

    #NRW
    e_p_nrw = 31.47
    g_p_nrw = 16.5

    #VIC
    e_p_vic = 31.12
    g_p_vic = 10.59    
    
    params['Source_el_price'] = 31.12/100 # [€/kWh]
    params['Source_gas_price'] = 10.59/100  # [€/kWh]
    params['Source_wh_T_price'] = 0.0078*0 # [€/(K*Kg/s)]

    params['Flow_pipe_length'] = 1000 # [m]

    n_con = 1000 # number of consumers

    for consumer_group in consumer_groups:
        # # Heat Pump Settings
        params[f'HP_C_{consumer_group}_dT_a_h'] = 3.0 # heat extraction - heating
        params[f'HP_C_{consumer_group}_dT_a_c'] = -3.0 # heat rejection - cooling
        params[f'HP_C_{consumer_group}_n'] = n_con
        params[f'HP_C_{consumer_group}_p_spec'] = 2000
        params[f'HP_C_{consumer_group}_p_fix'] = 0
        params[f'HP_C_{consumer_group}_T_amb'] = C2K(ambient_T)
        params[f'HP_C_{consumer_group}_eta_h'] = 0.55
        params[f'HP_C_{consumer_group}_eta_c'] = 0.5
        # BES settings
        #heat load
        params[f'BES_C_{consumer_group}_heating_Qdot'] = \
            data[f'heat_load'].apply(lambda x: x*1) *n_con 
        params[f'BES_C_{consumer_group}_heating_T_flow'] = C2K(45)
        #cool load
        params[f'BES_C_{consumer_group}_cooling_Qdot'] = \
            data[f'cool_load'].apply(lambda x: x*1) *n_con*-1
            # data[f'cool_load'] *n_con*-1
            
            
        params[f'BES_C_{consumer_group}_cooling_T_flow'] = C2K(7)
        # Resistance heating
        params[f'HS_el_C_{consumer_group}_n'] = n_con
        params[f'HS_el_C_{consumer_group}_p_spec'] = 100
        params[f'HS_el_C_{consumer_group}_p_fix'] = 0
        params[f'HS_el_C_{consumer_group}_efficiency'] = 0.01 # Resistance heating
        # Boiler
        params[f'HS_gas_C_{consumer_group}_n'] = n_con
        params[f'HS_gas_C_{consumer_group}_p_spec'] = 522
        params[f'HS_gas_C_{consumer_group}_p_fix'] = 10
        params[f'HS_gas_C_{consumer_group}_efficiency'] = 0.01 # Gas boiler


    
    # create Problem

    #This uses whole year \ disconnected days
    P = ES.create_problem(
        *make_tac_objective(ES, n=30, i=0.03),
        timesteps=timesteps,
        # scenarios=scenarios,
        data=params,
        name='heat_and_cool_whole_year'
    )

    # # This uses typical days and scenario
    # P = ES.create_problem(
    #     *make_tac_objective(ES, n=30, i=0.03),
    #     timesteps=timesteps,
    #     scenarios=scenarios,
    #     data=params,
    #     name='heat_and_cool_typ_4d'
    # )


    print()
    print(f'Problem has {P.num_cons} constraints and {P.num_vars} variables.')
    print()

    if validate_result:
        # set design variables according to design.csv
        result_df = pd.read_csv('design.csv', index_col='name')
        for dv in P.design_variables:
            dv.value = result_df.loc[dv.name]['value']
            dv.fix()
            print(f'set {dv.name} to {dv.value}')
        log_name = 'DESTEST_validation.log'
        design_name = 'design_validation.csv'
        operation_name = 'operation_validation.csv'
    else:
        log_name = 'DESTEST.log'
        design_name = 'design_2.csv'
        operation_name = 'operation_2.csv'

    m = to_pyomo(P)
    print('Solving...')
    options = dict(  # Options assuming Gurobi 9.1.1
        Seed=SEED,
        NonConvex=2,
        MIPGap=0.01,
        MIPFocus=1,
        LogFile=log_name,
        OutputFlag=1,
    )
    # m.computeIIS()
    # m.write('my_iis.ilp')
    # m.feasRelaxS(1, False, False, True)
    m.solve(solver='gurobi',options=options)

    print(P.design)
    print(P.operation)
    P.design.to_csv(design_name)
    P.operation.to_csv(operation_name)


    # tac = m.objVal
    tac = value(m.objective)
    print(f'\nExpected TAC {tac} €')
    h_demands = sum(p for p in P.parameters if p.name.endswith('heating_Qdot'))
    print
    c_demands = sum(p for p in P.parameters if p.name.endswith('cooling_Qdot'))
    tot_demand = P.weighted_sum(h_demands+c_demands, symbolic=False) / 1e3
    print(f'\nAnnual heating demand: {h_demands} MWh')
    print(f'\nAnnual cooling demand: {c_demands} MWh')
    print(f'\nAnnual energy demand: {tot_demand} MWh')
    print(f'\ncorresponds to: {tac / tot_demand} AUD/MWh')

    dir_path = pathlib.Path(__file__)
    os.chdir(dir_path.parent)
    now = datetime.now().strftime("%Y_%m_%d_%H_%M")
    with open(f'{now}_melb_results.pickle', 'wb') as f:
        pickle.dump((P), f)
    # plot(now,P)
    P.design.transpose().to_csv(design_name)
    P.operation.transpose().to_csv(operation_name)
    print(f'max value op: {P.operation.transpose().Source_wh_mflow.max()} kg/s')
    print(f'mean value op: {P.operation.transpose().Source_wh_mflow.mean()} kg/s')
    # P.operation.transpose()['']
    # P.data.getter().to_csv('letstry.csv')
    return ES, P, m

if __name__ == '__main__':
    import argparse
    import pathlib
    import os
    import sys
    # dir_path = pathlib.Path(__file__)
    # print(dir_path.parent)
    # os.chdir(dir_path.parent)
    os.chdir(sys.path[0])
    ap = argparse.ArgumentParser()

    ap.add_argument('-validate_result', '-vr', action='store_true',
                    default=False)

    if __package__ is None:
        import sys
        from pathlib import Path

        file = Path(__file__).resolve()
        sys.path.append(str(file.parents[1]))
    import Connect_melb

    __package__ = Connect_melb.__name__
    run_destest_case_study(**vars(ap.parse_args()))

    
