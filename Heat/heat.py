import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint


class MultiBodyHeat(object):
    def __init__(self, areas, common_areas, conductivities, blacknesses, cs, inner_heats, dt, steps_per_solve):
        self.num_of_parts = len(areas)
        self.areas = areas
        self.heat_connections = common_areas * conductivities
        self.blacknesses = blacknesses
        self.cs = cs
        self.inner_heats = inner_heats
        self.boltzmann = 5.67

        self.cur_temperatures = self.get_equilibrium()
        self.time_step = 0
        self.dt = dt
        self.steps_per_solve = steps_per_solve

    def heat_flow(self, temperatures, time):
        heat = np.zeros(self.num_of_parts)
        for part in range(self.num_of_parts):
            for another_part in range(self.num_of_parts):
                heat[part] += self.heat_connections[part, another_part] * (temperatures[part] - temperatures[another_part])
            heat[part] -= self.boltzmann * self.blacknesses[part] * self.areas[part] * (temperatures[part] / 100)**4
            heat[part] += self.inner_heats[part](time)
        return heat / self.cs

    def get_equilibrium(self):
        return fsolve(self.heat_flow, np.zeros(self.num_of_parts), args=(0,))

    def propagate(self):
        times = self.time_step*self.dt + np.arange(self.steps_per_solve)*self.dt
        self.time_step += self.steps_per_solve - 1
        res, info = odeint(self.heat_flow, self.cur_temperatures, times, full_output=True)
        self.cur_temperatures = res[-1]
        return self.cur_temperatures
