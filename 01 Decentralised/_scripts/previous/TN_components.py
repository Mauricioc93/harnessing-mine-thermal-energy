"""Component models for the low temperature district heating network."""
# This file is part of the COMANDO project which is released under the MIT
# license. See file LICENSE for full license details.
#
# AUTHORS: Dominik Hering, Marco Langiu
from comando.core import Component, BINARY, System, INTEGER
import numpy as np
from sympy import *
from tqdm import tgrange
from utilities import slope_intercept,segments_fit


### Additional functions
#piecewise linearization

def seg_power(fric,D_i,rho,eta_pump,m_max,n_seg):
    n_vect_x = 100 # discretisation (x) for function evaulation f(x)
    power_L_func = lambda m: (8*fric/(D_i**5*rho**2*np.pi**2)*(m)**3/1000)/eta_pump # cubic function P_el = f(mdot) kW
    m_vect = np.linspace(0,m_max, n_vect_x) #mdot discretisation
    power_vect = power_L_func(m_vect) #elec power evaluation
    px, py = segments_fit(m_vect, power_vect, n_seg) ## stepwise fitting
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

class BESMFlowTFixDT(Component):
    """A model for a building energy system.

    using mass flow and temperature connectors. Temperature difference is fixed

    Parameters
    ----------
    Qdot:
        The building's heat load, [kW]
    cp:
        Heating capacity, default=4.184 [kJ/kgK]
    dT:
        Temperature difference between t_flow and t_return, [K]
    t_flow:
        Flow temperature, [K]

    Connectors
    ----------
    IN_mflow:      Mass flow rate
    """

    def __init__(self, label):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        qdot = self.make_parameter('Qdot')  # [kW]
        # cp = self.make_parameter('cp', 4.184)
        # d_t = self.make_parameter('dT')
        self.make_parameter('T_flow')

        # m_flow = qdot / (d_t * cp)

        self.add_connector('IN_qdot', qdot)

class DummySource(Component):
    """A dummy resource that serves as a source for an arbitrary commodity.

    Parameters
    ----------
    price:
        float, defines the price per energy unit of the dummy source.

    Connectors
    ----------
    OUT:
        use of resource

    Expressions
    -----------
    variable_costs:
        Cost of for use
    """

    def __init__(self, label):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        price = self.make_parameter('price')

        #######################################################################
        # Operational Variables
        #######################################################################
        use = self.make_operational_variable('use', bounds=(0, None),
                                             init_val=0)
        self.add_output('OUT', use)

        self.add_expression('variable_costs', price * use)

class FreeMassflowSource(Component):
    """A resource that serves as a source for waste heat in T_flow and mass flow cond.

    Parameters
    ----------
    T_flow:
        Flow Temperature [K]
    T_return:
        return temperature of waste heat [K] - to be asssessed

    Operational Variables
    ---------------------
    m_flow
        Mass flow rate, [kg/s]

    Connectors
    ----------
    OUT_mflow:
        Outgoing mass flow
    """
    
    maxQ = 20 #[kW]
    max_mflow = 100
    def __init__(self, label):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        # Temperatures
        Tmax = self.make_parameter ('T_max', 273.15 + 100)
        Tmin = self.make_parameter ('T_min', 273.15 + 40)
        Tflow = self.make_design_variable('tflow', bounds=(273.15,273.15+30))
        self.add_le_constraint(Tflow,273.15+26, 'Tflow_set')
        T_price = self.make_parameter ('T_price', 0.01)
        # self.add_output(identifier='Out_Tout', expr=mflow)
        # Mass flow rate
        mflow = self.make_operational_variable('mflow', bounds=(0, None),init_val=0)
        self.add_output(identifier='OUT_mflow', expr=mflow)

        # cost per mflow temperature
        # self.add_expression('variable_costs', mflow * T_price * Tflow)
        # self.add_expression('variable_costs', mflow * T_price)

class HeatSourceDecentral(Component):
    """A model for a generic decentral heat source.

    Using mass flow and temperature connectors. Temperature difference is
    fixed.

    Parameters
    ----------
    cp:
        Heating capacity, default=4.184 [kJ/kgK]
    dT:
        Temperature difference between t_flow and t_return, [K]
    efficiency:
        Efficiency
    n:
        number of parallel units
    p_spec:
        Specific price of component  [€/kW]
    p_fix:
        Fix price of component [€]

    Design Variables
    ----------------
    b_build:
        Boolean for build decision of HS

    Connectors
    ----------
    OUT_mflow:
        Mass flow rate
    IN_P:
        Input for power consumption

    Expressions
    -----------
    investment_costs:    Investment cost of Heat source
    """

    Q_max = 20  # [kW]
    # mdot_max = 10  # [kg/s]

    def __init__(self, label):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        # cp = self.make_parameter('cp', 4.12)
        spec_price = self.make_parameter('p_spec', 100)  # €/kW
        fix_price = self.make_parameter('p_fix', 100)  # €
        # d_t = self.make_parameter('dT')
        eff = self.make_parameter('efficiency', value=1)
        n = self.make_parameter('n', value=1)

        qdot_design = self.make_design_variable('Qdot_max',
                                                bounds=(0, self.Q_max))
        b_build = self.make_design_variable('b_build', domain=BINARY)
        self.add_le_constraint(qdot_design, b_build * self.Q_max, 'Qdot_max')
        #######################################################################
        # Operational Variables
        #######################################################################
        q_dot = self.make_operational_variable('q_dot',
                                                bounds=(0, self.Q_max))
        self.add_le_constraint(q_dot, b_build * qdot_design, 'b_qdot')
        self.add_output('OUT_qdot', q_dot * n)
        p_in = q_dot
        self.add_le_constraint(p_in, qdot_design, 'p_max')
        self.add_input('IN_P', p_in * n / eff)

        # Investment costs
        inv_costs = (spec_price * qdot_design + b_build * fix_price) * n
        self.add_expression('investment_costs', inv_costs)

class NetworkPipe(Component):
    """A network component.
    MODIFY
    Collects and distributes energy flows. Temperature losses are calculated. 

    Parameters
    ----------
    U_avg:
        Average U value of network [W/K]
    cp:
        Heating capacity of medium, default=4.12 [kJ/kgK]
    Design_dT:
        Design temperature difference of network, default=15 [K]
    T_amb:
        Ambient air temperature. Is used to generate T_flow curve, [K]
    T_amb_design:
        Design ambient air temprature of lowest point in heating curve,
        default= -16 °C
    T_amb_tresh:
        Treshold temperature when no heating is required any more,
        default= 20 °C

    Design Variables
    ----------------
    b_{nodes}:
        Boolean decision variables for each possible pipe segment
    Return_Temp_max:
        Design Return temperature at T_amb_design [K]
    Return_Temp_min:
        Design Return temperature at T_amb_tresh [K]

    Operational Variables
    ---------------------
    mflow:
        Mass flow rate in network [kg/s]
    b_op:
        Boolean for operation (1 for On, 0 for off)
    T_in_a:
        Incoming Temperature at side a, [K]
    T_out_a:
        Outgoing Temperature at side a, [K]
    T_in_b:
        Incoming Temperature at side b, [K]
    T_out_b:
        Outgoing Temperature at side b, [K]

    Connectors
    ----------
    IN_mflow:
        Mass flow rate through network [kg/s]
    OUT_mflow:
        Mass flow rate through network [kg/s]

    Expressions
    -----------
    investment_costs:    Investment cost of pipes
    """

    pump_max = 10000 # [kW]
    def __init__(self, label, pipe_data):

        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        #pipe parameters
        u_avg = self.make_parameter('U_avg', 391)  # [W/mK] previous default, find values
        rho_pipe = self.make_parameter('rho_pipe', 8000) # [kg/m^3] - steel - find for PE
        cp_pipe = self.make_parameter('cp_pipe', 0.5) # [kJ/Kg/K] - steel - find for PE
        #water parameters
        cp = self.make_parameter('cp', 4.184) # [kJ/Kg/K] - water = 20° C
        rho = self.make_parameter('rho', 998) # [kg/m^3] - water = 20° C
        # Ground temperature
        t_gr = self.make_parameter('T_g',273.15+10)  # [K]
        length = self.make_parameter('length',100) # [m]
        
        # Pump efficiency
        eta_pump = self.make_parameter('eta_pump',0.72)

        #######################################################################
        # Design Variables
        #######################################################################
        # Flow temperature curve
        # t_net = self.make_design_variable('Network_T',
        #                                          bounds=(273.15, 373.15))

        inv_cost = pipe_data['Cost_m'].values[0]
        m_max = pipe_data['Max_mflow'].values[0]
        v_max = pipe_data['vmax'].values[0]
        D_i = pipe_data['Di'].values[0] # pipe inner diameter [m]
        fric = pipe_data['Friction_factor'].values[0] # friction factor
        Area = np.pi/4*D_i**2 # pipe inner area [m^2]
        Vol = Area*length
        # t_init = self.make_design_variable('t_init',bounds=(273.15,273.15+100))
        t_init = t_gr

        #######################################################################
        # Operational Variables
        #######################################################################
        m_flow = self.make_operational_variable("m_flow", bounds=(0, self.pump_max))
        b_op = self.make_operational_variable("b_op", domain=BINARY)
        self.add_le_constraint(m_flow, self.pump_max * b_op, 'bop_mflow')
        self.add_input(identifier='IN_mflow', expr=m_flow)
        self.add_output(identifier='OUT_mflow', expr=m_flow)


        t_in = self.make_operational_variable('t_in',
                                                bounds=(273.15, 373.15))

        # Temperature with losses
        t_out= self.make_operational_variable('t_out',
                                                 bounds=(273.15, 373.15))

        # thermal_cap = self.add_expression('thermal_cap',(rho*cp)*Area+(rho_pipe*cp_pipe)*Area/Vol)
        # thermal_delay = self.add_expression('thermal_delay',cp*m_flow*(t_out-t_in)/length)
        # thermal_loss = self.add_expression('thermal_loss',u_avg*Area/Vol*(t_out-t_gr))
        # t_out_change = (thermal_delay-thermal_loss)/thermal_cap
        # self.declare_state(t_out, t_out_change, t_init)
        self.add_eq_constraint(t_out,t_in,'constant_t')
        # self.add_eq_constraint(m_flow * cp,
        #                        (t_net - t_gr) * length * u_avg/1000,
        #                        'flow_losses')
        #circulation pump

        p_pump = self.make_operational_variable('P_pump', bounds=(0, self.pump_max))
        self.add_le_constraint(p_pump, self.pump_max * b_op, 'P_HP_op_c')
        n_seg = 20
        a_l,b_l = seg_power(fric,D_i,rho.value,eta_pump.value,m_max,n_seg)
        print(f'len of a = {len(a_l)} and b = {len(b_l)}')
        for i, (a,b) in enumerate(zip(a_l,b_l)):
            # if i == 0:
            self.add_le_constraint((m_flow*a+b)*length,p_pump,f'power_pump_{i}_{n_seg}')

        # circulation pump connected to the grid
        self.add_input(identifier='IN_P_pump', expr=p_pump)

        #######################################################################
        # Investment Costs
        #######################################################################
        inv_costs = length*pipe_data['Cost_m'] # Pipe length [m] * cost per metre [$/m] / 2 (1 of the 2 pipes)
        self.add_expression('investment_costs', inv_costs)

class CenHeatPump(Component): ### TEST with DETEST MODEL
    """Quadratic Heat pump model.

    Based on the following temperatures:
    - t_in_evap:
        Incoming temperature at evaporator [K]
    - t_out_evap:
        Outgoing temperature at evaporator [K]
    - t_out_cond:
        Outgoing temperature at condenser [K]
    - t_in_cond:
        Incoming temperature at condenser [K]

    Parameters
    ----------
    cop_nom : Float, COP efficiency.
        Determines the performance of the heat pump relative to Carnot in
        percent, default=0.6 ## modified to 0.4 according to Chiller efficiency in oemof - thermal
    n:
        Integer, Number of HP instances, default = 1
    cp:
        Heat capacity of fluid, default = 4.184 kJ/(kgK)

    Design Variables
    ----------------
    b_build:
        Boolean build decision of HP
    Qdot_design:
        Maximum heating capacity [kW]

    Operational Variables
    ---------------------
    b_op:
        Boolean for operation (1 for on, 0 for off)
    mflow_evap:
        Mass flow rate at evaporator [kg/s]
    mflow_cond:
        Mass flow rate at condenser [kg/s]
    Qdot_cond:
        Thermal power at condenser [kW]
    P_HP:
        Electrical power demand of HP [kW]

    Connectors
    ----------
    IN_mflow_evap:
        mass flow rate at evaporator [kg/s]
    IN_P:
        Electrical Power consumption [kW]
    OUT_mflow_cond:
        mass flow rate at Condenser [kg/s]

    Expressions
    -----------
        investment_costs:    Investment cost of HP
    """

    Q_max = 20  # [kW]
    mdot_max = 10  # [kg/s]

    def __init__(self, label):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        eta_nom_h = self.make_parameter('eta_h', 0.55) #heating efficiency
        eta_nom_c = self.make_parameter('eta_c', 0.5) # cooling efficiency
        cp = self.make_parameter('cp', 4.184)
        n = self.make_parameter('n', 1)
        # T_amb = self.make_parameter('T_amb',10+273.15)
        dT_a_h = self.make_parameter('dT_a_h',2.0)
        dT_a_c = self.make_parameter('dT_a_c',-2.0)
        dT_b_h = self.make_parameter('dT_b_h',-2.0)
        dT_b_c = self.make_parameter('dT_b_c',2.0)
        spec_price = self.make_parameter('p_spec', 500)  # €/kW
        fix_price = self.make_parameter('p_fix', 0)  # €

        #######################################################################
        # Design Variables
        #######################################################################
        qdot_design = self.make_design_variable('Qdot_design',
                                                bounds=(0, self.Q_max))
        b_build = self.make_design_variable('b_build', domain=BINARY,
                                            bounds=(0, 1))
        self.add_le_constraint(qdot_design, self.Q_max * b_build, 'HP_Q_max')

        #######################################################################
        # Operational Variable
        #######################################################################
        # if mode == 'reversible': #if not reversible, use other component
        if True:
            b_h = self.make_operational_variable('b_h', domain=BINARY)
            b_c = self.make_operational_variable('b_c',domain=BINARY)
            
            self.add_le_constraint(b_h, b_build, 'HP_heating_operation')
            self.add_le_constraint(b_c, b_build, 'HP_cooling_operation')
            self.add_le_constraint(b_c, 1 - b_h,'Heating_or_cooling')

            ######## temperature in side A #########
            # dT a
            t_in_a = self.make_operational_variable('t_in_a', bounds=(273.15,273.15+100))
            t_out_a_h = t_in_a - dT_a_h 
            t_out_a_c = t_in_a - dT_a_c 

            # dT b
            t_in_b = self.make_operational_variable('t_in_b', bounds=(273.15,273.15+100))
            t_out_b_h = t_in_b - dT_b_h 
            t_out_b_c = t_in_b - dT_b_c             

            ######## heating ########



            # Evaporator (side a)
            mflow_a_h = self.make_operational_variable('mflow_a_h',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_a_h, b_h * self.mdot_max,
                                'HP_mflow_a_h_op')
            self.add_le_constraint(mflow_a_h, b_build * self.mdot_max,
                                'HP_mflow_a_h_des')
            self.add_input(identifier='IN_mflow_a_h', expr=mflow_a_h * n)

            qdot_a_h = self.make_operational_variable('qdot_a_h',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_a_h, b_h * qdot_design,
                                'HP_qdot_a_h_op')
            self.add_le_constraint(qdot_a_h, b_build * qdot_design,
                                'HP_qdot_a_h_des')
            self.add_eq_constraint(qdot_a_h,mflow_a_h * (dT_a_h) * cp, 'heating_side_a_q')

            # Condenser (side b)
            #qdot
            qdot_b_h = self.make_operational_variable('qdot_b_h',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_b_h, b_h * qdot_design,
                                'HP_qdot_b_h_op')
            self.add_le_constraint(qdot_b_h, b_build * qdot_design,
                                'HP_qdot_b_h_des')
            self.add_output(identifier='OUT_qdot_h', expr=qdot_b_h * n)

            #mflow
            mflow_b_h = self.make_operational_variable('mflow_b_h',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_b_h, b_h * self.mdot_max,
                                'HP_mflow_b_h_op')
            self.add_le_constraint(mflow_b_h, b_build * self.mdot_max,
                                'HP_mflow_b_h_des')
            
            self.add_eq_constraint(qdot_b_h,mflow_b_h * (dT_b_h) * cp, 'qdot_mflow_b_heating')
            self.add_output(identifier='OUT_mflow_h', expr=mflow_b_h * n)

            # HP - heating

            self.add_le_constraint(qdot_b_h, self.Q_max * b_h, 'Qdot_HP_op_h')

            p_hp_h = self.make_operational_variable('P_HP_h', bounds=(0, self.Q_max))
            self.add_le_constraint(p_hp_h, self.Q_max * b_h, 'P_HP_op_h') 
            self.add_eq_constraint(p_hp_h * t_out_b_h * eta_nom_h, ## Careful with pinchpoint dT
                                (t_out_b_h - t_out_a_h) * qdot_b_h,
                                'input_output_relation_heating')

            # n_HP Heat pumps are connected to the grid in parallel
            self.add_input(identifier='IN_P_h', expr=p_hp_h * n)
            self.add_le_constraint(qdot_b_h, qdot_design, 'output_limit_h')

            self.add_eq_constraint(qdot_a_h + p_hp_h, qdot_b_h, 'energy_balance_h')

            ######## cooling ########

            # Condenser (side a)
            mflow_a_c = self.make_operational_variable('mflow_a_c',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_a_c, b_c * self.mdot_max,
                                'HP_mflow_a_c_op')
            self.add_le_constraint(mflow_a_c, b_build * self.mdot_max,
                                'HP_mflow_a_c_des')
            self.add_input(identifier='IN_mflow_a_c', expr=mflow_a_c * n)

            qdot_a_c = self.make_operational_variable('qdot_a_c',
                                                        bounds=(0,self.Q_max))
            self.add_le_constraint(qdot_a_c, self.Q_max * b_c, 'Qdot_HP_op_c')
            self.add_le_constraint(qdot_a_c, self.Q_max * b_build, 'Qdot_HP_des_c')
            self.add_eq_constraint(qdot_a_c,mflow_a_c * -1*(dT_a_c) * cp, 'cooling_side_a_c') # injected energy in outer fluid

            # Evaporator (side b)
            #qdot
            qdot_b_c = self.make_operational_variable('qdot_b_c',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_b_c, b_c * qdot_design,
                                'HP_qdot_b_c_op')
            self.add_le_constraint(qdot_b_c, b_build * qdot_design,
                                'HP_qdot_b_c_des')
            self.add_output(identifier='OUT_qdot_c', expr=qdot_b_c * n)

            #mflow
            mflow_b_c = self.make_operational_variable('mflow_b_c',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_b_c, b_c * self.mdot_max,
                                'HP_mflow_b_c_op')
            self.add_le_constraint(mflow_b_c, b_build * self.mdot_max,
                                'HP_mflow_b_c_des')
            
            self.add_eq_constraint(qdot_b_c,mflow_b_c * (dT_b_c) * cp, 'qdot_mflow_b_cooling')
            self.add_output(identifier='OUT_mflow_c', expr=mflow_b_c * n)

            # HP - cooling
            self.add_le_constraint(qdot_b_c, self.Q_max * b_c, 'Qdot_HP_op_c')
            self.add_le_constraint(qdot_b_c, self.Q_max * b_build, 'Qdot_HP_des_c')

            p_hp_c = self.make_operational_variable('P_HP_c', bounds=(0, self.Q_max))
            self.add_le_constraint(p_hp_c, self.Q_max * b_c, 'P_HP_op_c')
            self.add_eq_constraint(p_hp_c * t_out_b_c * eta_nom_c, ## Careful with pinchpoint dT
                                (t_out_a_c - t_out_b_c) * qdot_b_c, ## CONTINUE HERE
                                'input_output_relation_cooling')
                                
            # n_HP Heat pumps are connected to the grid in parallel
            self.add_input(identifier='IN_P_c', expr=p_hp_c * n)
            self.add_le_constraint(qdot_b_c, qdot_design, 'output_limit_c')
      
            self.add_eq_constraint(qdot_b_c + p_hp_c,qdot_a_c, 'energy_balance_c')

        #######################################################################
        # Misc
        #######################################################################
        # Investment costs
        inv_costs = (spec_price * qdot_design + b_build * fix_price) * n
        self.add_expression('investment_costs', inv_costs)


class DecenHeatPump(Component):
    """Quadratic Heat pump model.

    Based on the following temperatures:
    - t_in_evap:
        Incoming temperature at evaporator [K]
    - t_out_evap:
        Outgoing temperature at evaporator [K]
    - t_out_cond:
        Outgoing temperature at condenser [K]
    - t_in_cond:
        Incoming temperature at condenser [K]

    Parameters
    ----------
    cop_nom : Float, COP efficiency.
        Determines the performance of the heat pump relative to Carnot in
        percent, default=0.6 ## modified to 0.4 according to Chiller efficiency in oemof - thermal
    n:
        Integer, Number of HP instances, default = 1
    cp:
        Heat capacity of fluid, default = 4.184 kJ/(kgK)

    Design Variables
    ----------------
    b_build:
        Boolean build decision of HP
    Qdot_design:
        Maximum heating capacity [kW]

    Operational Variables
    ---------------------
    b_op:
        Boolean for operation (1 for on, 0 for off)
    mflow_evap:
        Mass flow rate at evaporator [kg/s]
    mflow_cond:
        Mass flow rate at condenser [kg/s]
    Qdot_cond:
        Thermal power at condenser [kW]
    P_HP:
        Electrical power demand of HP [kW]

    Connectors
    ----------
    IN_mflow_evap:
        mass flow rate at evaporator [kg/s]
    IN_P:
        Electrical Power consumption [kW]
    OUT_mflow_cond:
        mass flow rate at Condenser [kg/s]

    Expressions
    -----------
        investment_costs:    Investment cost of HP
    """

    Q_max = 20  # [kW]
    mdot_max = 10  # [kg/s]

    def __init__(self, label,
                 t_out_b_h,t_out_b_c):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        eta_nom_h = self.make_parameter('eta_h', 0.55) #heating efficiency
        eta_nom_c = self.make_parameter('eta_c', 0.50) # cooling efficiency
        cp = self.make_parameter('cp', 4.184)
        n = self.make_parameter('n', 1)
        # T_amb = self.make_parameter('T_amb',10+273.15)
        dT_a_h = self.make_parameter('dT_a_h',2.0)
        dT_a_c = self.make_parameter('dT_a_c',-2.0)
        # dT_b_h = self.make_parameter('dT_b_h',-2.0)
        # dT_b_c = self.make_parameter('dT_b_c',2.0)
        spec_price = self.make_parameter('p_spec', 100)  # €/kW
        fix_price = self.make_parameter('p_fix', 100)  # €

        #######################################################################
        # Design Variables
        #######################################################################
        qdot_design = self.make_design_variable('Qdot_design',
                                                bounds=(0, self.Q_max))
        b_build = self.make_design_variable('b_build', domain=BINARY,
                                            bounds=(0, 1))
        self.add_le_constraint(qdot_design, self.Q_max * b_build, 'HP_Q_max')

        #######################################################################
        # Operational Variable
        #######################################################################
        # if mode == 'reversible': #if not reversible, use other component
        if True:
            b_h = self.make_operational_variable('b_h', domain=BINARY)
            b_c = self.make_operational_variable('b_c',domain=BINARY)
            
            self.add_le_constraint(b_h, b_build, 'HP_heating_operation')
            self.add_le_constraint(b_c, b_build, 'HP_cooling_operation')
            self.add_le_constraint(b_c, 1 - b_h,'Heating_or_cooling')

            ######## temperature in side A #########
            # dT a
            t_in_a = self.make_operational_variable('t_in_a', bounds=(273.15,273.15+100))
            t_out_a_h = t_in_a - dT_a_h 
            t_out_a_c = t_in_a - dT_a_c 

       

            ######## heating ########



            # Evaporator (side a)
            mflow_a_h = self.make_operational_variable('mflow_a_h',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_a_h, b_h * self.mdot_max,
                                'HP_mflow_a_h_op')
            self.add_le_constraint(mflow_a_h, b_build * self.mdot_max,
                                'HP_mflow_a_h_des')
            self.add_input(identifier='IN_mflow_a_h', expr=mflow_a_h * n)

            qdot_a_h = self.make_operational_variable('qdot_a_h',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_a_h, b_h * qdot_design,
                                'HP_qdot_a_h_op')
            self.add_le_constraint(qdot_a_h, b_build * qdot_design,
                                'HP_qdot_a_h_des')
            self.add_eq_constraint(qdot_a_h,mflow_a_h * (dT_a_h) * cp, 'heating_side_a_q')

            # Condenser (side b)
            #qdot
            qdot_b_h = self.make_operational_variable('qdot_b_h',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_b_h, b_h * qdot_design,
                                'HP_qdot_b_h_op')
            self.add_le_constraint(qdot_b_h, b_build * qdot_design,
                                'HP_qdot_b_h_des')
            self.add_output(identifier='OUT_qdot_h', expr=qdot_b_h * n)


            # HP - heating

            self.add_le_constraint(qdot_b_h, self.Q_max * b_h, 'Qdot_HP_op_h')

            p_hp_h = self.make_operational_variable('P_HP_h', bounds=(0, self.Q_max))
            self.add_le_constraint(p_hp_h, self.Q_max * b_h, 'P_HP_op_h') 
            self.add_eq_constraint(p_hp_h * t_out_b_h * eta_nom_h, ## Careful with pinchpoint dT
                                (t_out_b_h - t_out_a_h) * qdot_b_h,
                                'input_output_relation_heating')

            # n_HP Heat pumps are connected to the grid in parallel
            self.add_input(identifier='IN_P_h', expr=p_hp_h * n)
            self.add_le_constraint(qdot_b_h, qdot_design, 'output_limit_h')

            self.add_eq_constraint(qdot_a_h + p_hp_h, qdot_b_h, 'energy_balance_h')

            ######## cooling ########

            # Condenser (side a)
            mflow_a_c = self.make_operational_variable('mflow_a_c',
                                                        bounds=(0, self.mdot_max))
            self.add_le_constraint(mflow_a_c, b_c * self.mdot_max,
                                'HP_mflow_a_c_op')
            self.add_le_constraint(mflow_a_c, b_build * self.mdot_max,
                                'HP_mflow_a_c_des')
            self.add_input(identifier='IN_mflow_a_c', expr=mflow_a_c * n)

            qdot_a_c = self.make_operational_variable('qdot_a_c',
                                                        bounds=(0,self.Q_max))
            self.add_le_constraint(qdot_a_c, self.Q_max * b_c, 'Qdot_HP_op_c')
            self.add_le_constraint(qdot_a_c, self.Q_max * b_build, 'Qdot_HP_des_c')
            self.add_eq_constraint(qdot_a_c,mflow_a_c * -1*(dT_a_c) * cp, 'cooling_side_a_c') # injected energy in outer fluid

            # Evaporator (side b)
            #qdot
            qdot_b_c = self.make_operational_variable('qdot_b_c',
                                                        bounds=(0, self.Q_max))
            self.add_le_constraint(qdot_b_c, b_c * qdot_design,
                                'HP_qdot_b_c_op')
            self.add_le_constraint(qdot_b_c, b_build * qdot_design,
                                'HP_qdot_b_c_des')
            self.add_output(identifier='OUT_qdot_c', expr=qdot_b_c * n)

            # HP - cooling
            self.add_le_constraint(qdot_b_c, self.Q_max * b_c, 'Qdot_HP_op_c')
            self.add_le_constraint(qdot_b_c, self.Q_max * b_build, 'Qdot_HP_des_c')

            p_hp_c = self.make_operational_variable('P_HP_c', bounds=(0, self.Q_max))
            self.add_le_constraint(p_hp_c, self.Q_max * b_c, 'P_HP_op_c')
            self.add_eq_constraint(p_hp_c * t_out_b_c * eta_nom_c, ## Careful with pinchpoint dT
                                (t_out_a_c - t_out_b_c) * qdot_b_c, ## CONTINUE HERE
                                'input_output_relation_cooling')
                                
            # n_HP Heat pumps are connected to the grid in parallel
            self.add_input(identifier='IN_P_c', expr=p_hp_c * n)
            self.add_le_constraint(qdot_b_c, qdot_design, 'output_limit_c')
      
            self.add_eq_constraint(qdot_b_c + p_hp_c,qdot_a_c, 'energy_balance_c')

        #######################################################################
        # Misc
        #######################################################################
        # Investment costs
        inv_costs = (spec_price * qdot_design + b_build * fix_price) * n
        self.add_expression('investment_costs', inv_costs)

class HeatExchanger(Component):
    """Heat exchanger model.

    Based on the following temperatures:
    - t_in_a:
        Incoming temperature at side a [K]
    - t_out_a:
        Outgoing temperature at side a [K]
    - t_out_b:
        Outgoing temperature at side b [K]
    - t_in_b:
        Incoming temperature at side b [K]

    Parameters
    ----------
    n:
        Integer, Number of HP instances, default = 1
    cp:
        Heat capacity of fluid, default = 4.184 kJ/(kgK)

    Design Variables
    ----------------
    b_build:
        Boolean for build decision of HX
    Qdot_design:
        Maximum heating capacity [kW]

    Operational Variables
    ---------------------
    b_op:
        Boolean operational variable.
        b_op=0 for no mass flow and no temperature constraints
        b_op=1 for heat exchanger operation
    mflow_a:
        Incoming mass flow rate at side a [kg/s]
    mflow_b:
        Outgoing mass flow rate at side b [kg/s]

    Connectors
    ----------
    IN_mflow_a:
        mass flow rate at side a [kg/s]
    OUT_mflow_b:
        mass flow rate at side b [kg/s]

    Expressions
    -----------
        investment_costs:    Investment cost of HX
    """

    Q_max = 400  # [kW]
    mdot_max = 10  # [kg/s]

    def __init__(self, label, b_connect=None, b_build_hp=None,
                 t_in_a=None, t_out_a=None, t_in_b=None, t_out_b=None):
        super().__init__(label)
        #######################################################################
        # Parameters
        #######################################################################
        cp = self.make_parameter('cp', 4.184)
        n = self.make_parameter('n', 1)

        #######################################################################
        # Design Variables
        #######################################################################
        qdot_design = self.make_design_variable('Qdot_design',
                                                bounds=(0, self.Q_max))
        b_build = self.make_design_variable('b_build', domain=BINARY)
        if b_build_hp is not None:
            # Opposite of HP build decision
            self.add_le_constraint(b_build, 1 - b_build_hp)
        self.add_le_constraint(qdot_design, self.Q_max * b_build, 'Qdot_max')
        if b_connect is not None:
            self.add_le_constraint(b_build, b_connect, 'connection')

        #######################################################################
        # Operational Variable
        #######################################################################
        b_op = self.make_operational_variable('b_op', domain=BINARY)
        self.add_le_constraint(b_op, b_build, 'operation')
        # Side a
        mflow_a = self.make_operational_variable('mflow_a',
                                                 bounds=(0, self.mdot_max))
        self.add_le_constraint(mflow_a, b_op * self.mdot_max, 'b_hx_mflow_a')
        self.add_le_constraint(mflow_a, b_build * self.mdot_max,
                               'bbuild_hx_mflow_a')
        self.add_input(identifier='IN_mflow_a', expr=mflow_a * n)

        qdot_in = mflow_a * (t_in_a - t_out_a) * cp

        # Side b
        mflow_b = self.make_operational_variable('mflow_b',
                                                 bounds=(0, self.mdot_max))
        self.add_le_constraint(mflow_b, b_op * self.mdot_max, 'b_hx_mflow_b')
        self.add_le_constraint(mflow_b, b_build * self.mdot_max,
                               'bbuild_hx_mflow_b')
        self.add_output(identifier='OUT_mflow_b', expr=mflow_b * n)

        qdot_out = mflow_b * cp * (t_out_b - t_in_b)

        self.add_le_constraint(qdot_out, qdot_design, 'design_Limit')
        self.add_le_constraint(t_out_b * b_op, t_in_a * b_op,
                               'max_temp_increase')

        self.add_eq_constraint(qdot_in, qdot_out, 'heat_flow')

        #######################################################################
        # Misc
        #######################################################################
        # Investment costs
        inv_costs = 90 * qdot_design * n
        self.add_expression('investment_costs', inv_costs)

class LinkingComponent(System):
    """Create a linking component, modeled as a COMANDO system.

    This component contains a Heat Pump and a Heat Exchanger model.
    """

    def __init__(self, label, t_in_a, t_out_a,
                 t_in_b, t_out_b, b_connect=None):
        super().__init__(label)
        hp = HeatPump(f'HP_{label}', b_connect=b_connect,
                      t_in_evap=t_in_a, t_out_evap=t_out_a,
                      t_in_cond=t_in_b, t_out_cond=t_out_b)
        hx = HeatExchanger(f'HX_{label}', b_build_hp=hp['b_build'],
                           t_in_a=t_in_a, t_out_a=t_out_a,
                           t_in_b=t_in_b, t_out_b=t_out_b,
                           b_connect=b_connect)
        for comp in [hp, hx]:
            self.add(comp)
        # Side a
        self.connect('IN_mflow_a', [hp.IN_mflow_evap,
                                    hx.IN_mflow_a])
        # Side b
        self.connect('OUT_mflow_b', [hp.OUT_mflow_cond,
                                     hx.OUT_mflow_b])

        self.expose_connector(hp.IN_P, 'IN_P')

class Consumer(System):
    """Create a consumer group, modeled as a COMANDO system.

    This component contains a linking component, two heat sources and one
    demand.
    """

    def __init__(self, label):
        super().__init__(label)
        hs_el = HeatSourceDecentral(f'HS_el_{label}')
        hs_gas = HeatSourceDecentral(f'HS_gas_{label}')
        self.add_le_constraint(hs_el['b_build'] + hs_gas['b_build'], 1)


        building_h = BESMFlowTFixDT(f'BES_{label}_heating')
        building_c = BESMFlowTFixDT(f'BES_{label}_cooling')
        hp = DecenHeatPump(f'HP_{label}', #Check here
            t_out_b_h=building_h['T_flow'],t_out_b_c=building_c['T_flow']) #Check here

        for comp in [hp, hs_el, hs_gas, building_h,building_c]:
            self.add(comp)

        
        self.connect('qdot_to_bes_h', [hp.OUT_qdot_h, # Check here
                                      hs_el.OUT_qdot,
                                      hs_gas.OUT_qdot,
                                      building_h.IN_qdot])

        self.connect('qdot_to_bes_c', [hp.OUT_qdot_c, # Check here
                                    building_c.IN_qdot])

        self.connect('IN_P_el', [hp.IN_P_c,
                                 hp.IN_P_h,
                                 hs_el.IN_P,
                                 hs_el.IN_P])

        self.expose_connector(hs_gas.IN_P, 'IN_P_gas')
        self.expose_connector(hp.IN_mflow_a_h, 'IN_mflow_a_h')
        self.expose_connector(hp.IN_mflow_a_c, 'IN_mflow_a_c')