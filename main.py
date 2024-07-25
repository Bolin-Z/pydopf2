import pandapower as pp
from pandapower import converter
from pandapower import pandapowerNet
import numpy as np
import matplotlib.pyplot as plt
from random import uniform, seed
from copy import copy, deepcopy
from math import sin, cos
from abc import ABC, abstractmethod

PARAM = {
    # PSO
    "swarm_size" : 20,
    "v_percent"  : 0.2,
    "c1"         : 1.415,
    "c2"         : 1.415,
    "w"          : 0.9,
    "MAX_G"      : 60,
    # PF
    "LAMBDA_1"   : 0.7922,
    "ETA_1"      : 0.7318,
    "LAMBDA_2"   : 0.2078,
    "ETA_2"      : 0.2682,
    "EPSILON_IN" : 1 * 10 ** (-5),
    "EPSILON_OUT": 1 * 10 ** (-4),
    "TIE_LINES_Z": [0.053 + 0.057j],
    # OTHER
    "PENALTY_ON" : True,
    "PENALTY_G"  : 5,
    "PENALTY_V"  : 50
}

def cosd(degree):
    return np.cos(np.radians(degree))

def sind(degree):
    return np.sin(np.radians(degree))

class TieLineInfo:
    def __init__(self, name:str, from_bus:int, to_bus:int, impedance:complex) -> None:
        # basic information
        self.name = name
        self.from_bus = from_bus
        self.to_bus = to_bus
        self.impedance = impedance
        # state information
        self.inject_power = 0j
        self.from_vm = 0.0
        self.from_va = 0.0
        self.to_vm = 0.0
        self.to_va = 0.0

class Particle:
    def __init__(self, dimension, tie_lines:dict[str, TieLineInfo]) -> None:
        self.dim = dimension
        self.x = [0.0 for _ in range(self.dim)]
        self.v = [0.0 for _ in range(self.dim)]
        self.fitness = float('inf')
        self.pbest = [0.0 for _ in range(self.dim)]
        self.pbest_fitness = float('inf')
        # cost
        self.x_cost = float('inf')
        self.pbest_cost = float('inf')
        # network state information
        self.x_tie_lines = deepcopy(tie_lines)
        self.pbest_tie_lines = deepcopy(tie_lines)

class Agent(ABC):
    def __init__(self, net:pandapowerNet, tie_lines:dict[str, TieLineInfo]) -> None:
        self.net = net
        self.x_upper_bound = []
        self.x_lower_bound = []

        self.num_of_generator = len(net.gen)
        self.x_upper_bound.extend(list(net.gen.max_p_mw.values))
        self.x_lower_bound.extend(list(net.gen.min_p_mw.values))
        
        mask = list(net.gen.bus.values)
        mask.extend(list(net.ext_grid.bus.values))
        self.x_upper_bound.extend(list(net.bus.max_vm_pu.loc[mask].values))
        self.x_lower_bound.extend(list(net.bus.min_vm_pu.loc[mask].values))
        self.PQ_bus = [i for i in range(len(self.net.bus)) if i not in mask]

        self.dim = len(self.x_upper_bound)
        x_range = [self.x_upper_bound[i] - self.x_lower_bound[i] for i in range(self.dim)]
        self.v_lower_bound = [- d * PARAM["v_percent"] for d in x_range]
        self.v_upper_bound = [d * PARAM["v_percent"] for d in x_range]

        self.total_loads_p_mw = sum(net.load.p_mw._values)
        self.total_loads_q_mvar = sum(net.load.q_mvar.values)

        self.tie_lines = deepcopy(tie_lines)
        self.num_tie_lines = len(self.tie_lines)
        self.virtual_sgens = {name : pp.create_sgen(self.net, info.to_bus, 0) for name, info in self.tie_lines.items()}
        self.virtual_loads = {name : pp.create_load(self.net, info.to_bus, 0) for name, info in self.tie_lines.items()}

        # Initialize Swarm
        self.gbest_idx = 0
        self.swarm = [Particle(self.dim, self.tie_lines) for _ in range(PARAM["swarm_size"])]
        for i in range(PARAM["swarm_size"]):
            p = self.swarm[i]
            for d in range(p.dim):
                p.x[d] = uniform(self.x_lower_bound[d], self.x_upper_bound[d])
                p.v[d] = uniform(self.v_lower_bound[d], self.v_upper_bound[d])
                p.pbest[d] = p.x[d]
            self.set_control_variable(p.x)
            try:
                pp.runpp(self.net, tolerance_mva=PARAM["EPSILON_IN"], numba=False)
                p.x_cost = self.total_cost(self.net)
                p.fitness = p.x_cost + self.penalty(self.net)
                for line in p.x_tie_lines.values():
                    p.x_tie_lines[line.name].from_vm = self.net.res_bus.vm_pu.at[line.from_bus]
                    p.x_tie_lines[line.name].from_va = self.net.res_bus.va_degree.at[line.from_bus]
                    p.x_tie_lines[line.name].to_vm = self.net.res_bus.vm_pu.at[line.to_bus]
                    p.x_tie_lines[line.name].to_va = self.net.res_bus.va_degree.at[line.to_bus]
            except:
                p.x_cost = float('inf')
                p.fitness = float('inf')
            p.pbest_cost = p.x_cost
            p.pbest_fitness = p.fitness
            p.pbest_tie_lines = deepcopy(p.x_tie_lines)
            if p.pbest_fitness < self.swarm[self.gbest_idx].pbest_fitness:
                self.gbest_idx = i
    
    def evolve(self) -> None:
        for i in range(PARAM["swarm_size"]):
            gBest = self.swarm[self.gbest_idx]
            p = self.swarm[i]
            for d in range(self.dim):
                # learning phase
                p.v[d] = PARAM["w"] * p.v[d] + PARAM["c1"] * uniform(0, 1) * (p.pbest[d] - p.x[d]) \
                        + PARAM["c2"] * uniform(0, 1) * (gBest.pbest[d] - p.x[d])
                p.v[d] = max(self.v_lower_bound[d], min(p.v[d], self.v_upper_bound[d]))
                p.x[d] = p.x[d] + p.v[d]
                p.x[d] = max(self.x_lower_bound[d], min(p.x[d], self.x_upper_bound[d]))
            # evaluation
            self.set_control_variable(p.x)
            self.set_injection_power(p.x_tie_lines)
            try:
                # power flow converge
                pp.runpp(self.net, tolerance_mva=PARAM["EPSILON_IN"], numba=False)
                p.x_cost = self.total_cost(self.net)
                p.fitness = p.x_cost + self.penalty(self.net)
                for line in p.x_tie_lines.values():
                    p.x_tie_lines[line.name].from_vm = self.net.res_bus.vm_pu.at[line.from_bus]
                    p.x_tie_lines[line.name].from_va = self.net.res_bus.va_degree.at[line.from_bus]
                    p.x_tie_lines[line.name].to_vm = self.net.res_bus.vm_pu.at[line.to_bus]
                    p.x_tie_lines[line.name].to_va = self.net.res_bus.va_degree.at[line.to_bus]
            except:
                # power flow does not converge
                p.x_cost = float("inf")
                p.fitness = float("inf")
            # update local best
            if p.fitness < p.pbest_fitness:
                p.pbest = copy(p.x)
                p.pbest_cost = p.x_cost
                p.pbest_fitness = p.fitness
                p.pbest_tie_lines = deepcopy(p.x_tie_lines)
                # update global best
                if p.pbest_fitness < gBest.pbest_fitness:
                    self.gbest_idx = i

    @abstractmethod
    def total_cost(self, net:pandapowerNet) -> float:
        ...

    def penalty(self, net:pandapowerNet) -> float:
        penalty = 0.0
        if PARAM["PENALTY_ON"]:
            # SLACK BUS P
            slack_bus_p_mw = net.res_ext_grid.p_mw.at[0]
            if slack_bus_p_mw > net.ext_grid.max_p_mw.at[0]:
                penalty += (slack_bus_p_mw - net.ext_grid.max_p_mw.at[0]) * PARAM["PENALTY_G"]
            elif slack_bus_p_mw < net.ext_grid.min_p_mw.at[0]:
                penalty += (net.ext_grid.min_p_mw.at[0] - slack_bus_p_mw) * PARAM["PENALTY_G"]
            # PV BUS Q
            for i in range(self.num_of_generator):
                gen_q_mvar = net.res_gen.q_mvar.at[i]
                if gen_q_mvar > net.gen.max_q_mvar.at[i]:
                    penalty += (gen_q_mvar - net.gen.max_q_mvar.at[i]) * PARAM["PENALTY_G"]
                elif gen_q_mvar < net.gen.min_q_mvar.at[i]:
                    penalty += (net.gen.min_q_mvar.at[i] - gen_q_mvar) * PARAM["PENALTY_G"]
            # PQ BUS V
            for idx in self.PQ_bus:
                pq_bus_v = net.res_bus.vm_pu.at[idx]
                if pq_bus_v > net.bus.max_vm_pu.at[idx]:
                    penalty += (pq_bus_v - net.bus.max_vm_pu.at[idx]) * PARAM["PENALTY_V"]
                elif pq_bus_v < net.bus.min_vm_pu.at[idx]:
                    penalty += (net.bus.min_vm_pu.at[idx] - pq_bus_v) * PARAM["PENALTY_V"]
        
        return penalty

    def set_control_variable(self, x:list[float]) -> None:
        self.net.gen.p_mw = np.array(x[0 : self.num_of_generator])
        self.net.gen.vm_pu = np.array(x[self.num_of_generator : 2 * self.num_of_generator])
        self.net.ext_grid.vm_pu = np.array(x[2 * self.num_of_generator])
    
    def set_injection_power(self, lines:dict[str, TieLineInfo]) -> None:
        for line in lines.values():
            p = line.inject_power
            if p.real < 0:
                self.net.load.p_mw.at[self.virtual_loads[line.name]] = - p.real
                self.net.sgen.p_mw.at[self.virtual_sgens[line.name]] = 0
            else:
                self.net.sgen.p_mw.at[self.virtual_sgens[line.name]] = p.real
                self.net.load.p_mw.at[self.virtual_loads[line.name]] = 0
            self.net.sgen.q_mvar.at[self.virtual_sgens[line.name]] = p.imag

class Agent9(Agent):
    def total_cost(self, net: pandapowerNet) -> float:
        cost = 0.0
        p_mw = []
        p_mw.append(net.res_ext_grid.p_mw.at[0])
        p_mw.extend(list(net.res_gen.p_mw.values))
        for i in range(len(p_mw)):
            if p_mw[i] > 0:
                cost += net.poly_cost.cp0_eur.at[i] + \
                    net.poly_cost.cp1_eur_per_mw.at[i] * p_mw[i] + \
                        net.poly_cost.cp2_eur_per_mw2.at[i] * (p_mw[i] ** 2)
        return cost
    
    def penalty(self, net: pandapowerNet) -> float:
        return super().penalty(net)


class Agent30(Agent):
    def total_cost(self, net: pandapowerNet) -> float:
        # factor values
        a = [0., 0., 0., 0., 0., 0.]
        b = [2, 1.75, 1., 3.25, 3, 3]
        c = [0.00375, 0.0175, 0.0625, 0.00834, 0.025, 0.025]
        e = [18, 16, 14, 12, 13, 13.5]
        f = [0.037, 0.038, 0.04, 0.045, 0.042, 0.041]  
        cost = 0.0
        p_mw = []
        p_mw.append(net.res_ext_grid.p_mw.at[0])
        p_mw.extend(list(net.res_gen.p_mw.values))
        min_p_mw = []
        min_p_mw.append(net.ext_grid.min_p_mw.at[0])
        min_p_mw.extend(net.gen.min_p_mw.values)
        for i in range(len(p_mw)):
            if p_mw[i] > 0:
                cost += a[i] + b[i] * p_mw[i] + c[i] * (p_mw[i] ** 2) + abs(e[i] * sin(f[i] * (min_p_mw[i] - p_mw[i])))
        return cost
    
    def penalty(self, net: pandapowerNet) -> float:
        return super().penalty(net)

if __name__ == "__main__":
    Net_case_9 = converter.from_mpc("./power_case/case9_1.m")
    Net_case_30 = converter.from_mpc("./power_case/case_ieee30_1.m")
    # Modified Network
    Net_case_9.bus.max_vm_pu = np.array([1.10 for _ in range(len(Net_case_9.bus))])
    Net_case_9.bus.min_vm_pu = np.array([0.9  for _ in range(len(Net_case_9.bus))])

    Net_case_30.gen.max_p_mw = np.array([80, 50, 35, 30, 40])
    Net_case_30.gen.min_p_mw = np.array([20, 15, 10, 10, 12])
    Net_case_30.ext_grid.max_p_mw = np.array([250])
    Net_case_30.ext_grid.min_p_mw = np.array([50])
    Net_case_30.gen.max_q_mvar = np.array([60, 62.45, 48.73, 40, 44.72])
    Net_case_30.gen.min_q_mvar = np.array([-20, -15, -15, -10, -15])
    Net_case_30.ext_grid.max_q_mvar = np.array([150])
    Net_case_30.ext_grid.min_q_mvar = np.array([-20])

    net9 = Agent9(
        Net_case_9,
        {
            "Line-1" : TieLineInfo("Line-1", 3, 9, 0.053 + 0.057j)
        }
    )
    net30 = Agent30(
        Net_case_30,
        {
            "Line-1" : TieLineInfo("Line-1", 2, 30, 0.053 + 0.057j)
        }
    )

    OUTER_LOOP = 1
    INNER_LOOP = 20

    net9_fitness = []
    net9_cost = []
    net30_fitness = []
    net30_cost = []
    net9_gbest_idx = []
    net30_gbest_idx = []
    
    net9_fitness.append(net9.swarm[net9.gbest_idx].pbest_fitness)
    net9_cost.append(net9.swarm[net9.gbest_idx].pbest_cost)
    net9_gbest_idx.append(net9.gbest_idx)
    net30_fitness.append(net30.swarm[net30.gbest_idx].pbest_fitness)
    net30_cost.append(net30.swarm[net30.gbest_idx].pbest_cost)
    net30_gbest_idx.append(net30.gbest_idx)   


    for _ in range(OUTER_LOOP):
        for _ in range(INNER_LOOP):
            print(f"round {_}")
            # Net-9 local optimization
            net9.evolve()
            # Net-30 local optimization
            net30.evolve()

            net9_fitness.append(net9.swarm[net9.gbest_idx].pbest_fitness)
            net9_cost.append(net9.swarm[net9.gbest_idx].pbest_cost)
            net9_gbest_idx.append(net9.gbest_idx)
            net30_fitness.append(net30.swarm[net30.gbest_idx].pbest_fitness)
            net30_cost.append(net30.swarm[net30.gbest_idx].pbest_cost)
            net30_gbest_idx.append(net30.gbest_idx)   
        # communication
        pass

    X = np.arange(len(net9_fitness))
    F_9 = np.array(net9_fitness)
    C_9 = np.array(net9_cost)
    I_9 = np.array(net9_gbest_idx)
    F_30 = np.array(net30_fitness)
    C_30 = np.array(net30_cost)
    I_30 = np.array(net30_gbest_idx)
    TOTAL_C = C_9 + C_30
    TOTAL_F = F_9 + F_30

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(X, F_9, color="b", label="f9")
    ax.plot(X, C_9, color="r", label="c9")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(X, F_30, color="g", label="f30")
    ax.plot(X, C_30, color="c", label="c30")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(X, TOTAL_C, color="r", label="total_c")
    ax.plot(X, TOTAL_F, color="b", label="total_f")
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(X, I_9, color="b", marker="D")
    plt.show()

    fig, ax = plt.subplots()
    ax.grid()
    ax.scatter(X, I_30, color="r", marker="D")
    plt.show()
