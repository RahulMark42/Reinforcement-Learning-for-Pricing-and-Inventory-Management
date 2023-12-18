import numpy as np

class CumSumList:
    def __init__(self, elements):
        self.cumsums = [0] * len(elements)
        partial_sum = 0
        for i, element in enumerate(elements):
            partial_sum += element
            self.cumsums[i] = partial_sum

    def sum_between(self, i, j):
        if i > j:
            return 0
        top = self.cumsums[j] if j < len(self.cumsums) else self.cumsums[-1]
        low = self.cumsums[i - 1] if i >= 1 else 0
        return top - low

class WagnerWhitin:
    def __init__(self, demand):
        self.demand = demand
        self.capacity = 50
        self.holding_cost = 3
        self.unit_price = 10
        self.fixed_order_cost = 50
        self.variable_order_cost = 10
        self.lead_time = 2
        self.periods = len(demand)

    def calculate_profit(self):
        order_costs = [self.fixed_order_cost + self.variable_order_cost]*self.periods
        holding_costs = [self.holding_cost]*self.periods
        res = wagner_whitin(self.demand, order_costs=order_costs, holding_costs=holding_costs)
        order_qtys = res['solution']
        cost = res['cost']
        price = 0
        indices_greater_than_zero = [index for index, element in enumerate(order_qtys) if element > 0]
        for i in indices_greater_than_zero:
            price += order_qtys[i]*self.unit_price
        return price - int(cost)

class LotForLot:
    def __init__(self, demand):
        self.demand = demand
        self.capacity = 50
        self.holding_cost = 3
        self.unit_price = 20
        self.fixed_order_cost = 50
        self.variable_order_cost = 10
        self.lead_time = 2
        self.num_periods = len(demand)
        self.inventory_levels = [0] * (self.num_periods + 1)
        self.order_quantities = [0] * self.num_periods
        self.total_profit = 0

    def run(self):
        order_costs = (self.fixed_order_cost + self.fixed_order_cost)*self.num_periods
        holding_costs = ()

def generate_random_demand():
    demand_hist = []
    for i in range(52):
        for j in range(4):
            random_demand = np.random.normal(3, 1.5)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
        random_demand = np.random.normal(6, 1)
        if random_demand < 0:
            random_demand = 0
        random_demand = np.round(random_demand)
        demand_hist.append(random_demand)
        for j in range(2):
            random_demand = np.random.normal(12, 2)
            if random_demand < 0:
                random_demand = 0
            random_demand = np.round(random_demand)
            demand_hist.append(random_demand)
    return demand_hist

def profit_calculation_sS(s,S,demand_records):
    total_profit = 0
    inv_level = 25 # inventory on hand, use this to calculate inventory costs
    lead_time = 2
    capacity = 50
    holding_cost = 3
    fixed_order_cost = 50
    variable_order_cost = 10
    unit_price = 30
    order_arrival_list = []
    inv_level_day = []
    for current_period in range(len(demand_records)):
        inv_pos = inv_level
        if len(order_arrival_list) > 0:
            for i in range(len(order_arrival_list)):
                inv_pos += order_arrival_list[i][1]
        if inv_pos <= s:
            order_quantity = min(20,S-inv_pos)
            order_arrival_list.append([current_period+lead_time, order_quantity])
            y = 1
        else:
            order_quantity = 0
            y = 0
        if len(order_arrival_list) > 0:
            if current_period == order_arrival_list[0][0]:
                inv_level = min(capacity, inv_level + order_arrival_list[0][1])
                order_arrival_list.pop(0)
        demand = demand_records[current_period]
        units_sold = demand if demand <= inv_level else inv_level
        profit = units_sold*unit_price-holding_cost*inv_level-y*fixed_order_cost-order_quantity*variable_order_cost
        inv_level = max(0,inv_level-demand)
        total_profit += profit
        inv_level_day.append(inv_level)
    return total_profit, inv_level_day

def generate_multiple_test_scenarios():
  demand_test = []
  for k in range(100,200):
      np.random.seed(k)
      demand_future = []
      for i in range(52):
          for j in range(4):
              random_demand = np.random.normal(3, 1.5)
              if random_demand < 0:
                  random_demand = 0
              random_demand = np.round(random_demand)
              demand_future.append(random_demand)
          random_demand = np.random.normal(6, 1)
          if random_demand < 0:
              random_demand = 0
          random_demand = np.round(random_demand)
          demand_future.append(random_demand)
          for j in range(2):
              random_demand = np.random.normal(12, 2)
              if random_demand < 0:
                  random_demand = 0
              random_demand = np.round(random_demand)
              demand_future.append(random_demand)
      demand_test.append(demand_future)
  return demand_test

def wagner_whitin(demands, order_costs, holding_costs):
    d, o, h = demands, order_costs, holding_costs
    #_validate_inputs(demands, order_costs, holding_costs)
    d_cumsum = CumSumList(d)
    assert d[-1] > 0, "Final demand should be positive"

    T = len(d)  
    F = {-1: 0}  
    t_star_star = 0  

    cover_by = {}  
    for t in range(len(demands)):
        if d[t] == 0:
            F[t] = F[t - 1]
            cover_by[t] = t
            continue
        assert d[t] > 0

        S_t = 0 
        min_args = []  
        for j in reversed(range(t_star_star, t + 1)):
            S_t += h[j] * d_cumsum.sum_between(j + 1, t)
            min_args.append(o[j] + S_t + F[j - 1])

        argmin = min_args.index(min(min_args))

        t_star_star = max(t_star_star, t - argmin)

        F[t] = min_args[argmin]
        cover_by[t] = t - argmin

    t = T - 1
    solution = [0] * T
    while True:
        j = cover_by[t]  
        solution[j] = sum(d[j : t + 1])
        t = j - 1 
        if j == 0:
            break 
    return {"cost": F[len(demands) - 1], "solution": solution}


