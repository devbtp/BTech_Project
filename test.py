import numpy as np

def get_us_and_vs(bfs, costs):
    us = [None] * len(costs)
    vs = [None] * len(costs[0])
    us[0] = 0
    bfs_copy = bfs.copy()
    while len(bfs_copy) > 0:
        for index, bv in enumerate(bfs_copy):
            i, j = bv[0]
            if us[i] is None and vs[j] is None: continue

            cost = costs[i][j]
            if us[i] is None:
                us[i] = cost - vs[j]
            else:
                vs[j] = cost - us[i]
            bfs_copy.pop(index)
            break

    return us, vs


def vam(supply, demand):
    from collections import defaultdict

    costs = {
        'X': {'A': 2, 'B': 2, 'C': 2, 'D': 1},
        'Y': {'A': 10, 'B': 8, 'C': 5, 'D': 4},
        'Z': {'A': 7, 'B': 6, 'C': 6, 'D': 8}}
    idemand = {'A': 40, 'B': 30, 'C': 40, 'D': 40}
    cols = sorted(idemand.keys())

    isupply = {'X': 30, 'Y': 70, 'Z': 50}
    res = dict((k, defaultdict(int)) for k in costs)

    g = {}
    for x in isupply:
        g[x] = sorted(costs[x].keys(), key=lambda g: costs[x][g])
    for x in idemand:
        g[x] = sorted(costs.keys(), key=lambda g: costs[g][x])

    while g:
        d = {}
        for x in idemand:
            d[x] = (costs[g[x][1]][x] - costs[g[x][0]][x]) if len(g[x]) > 1 else costs[g[x][0]][x]
        s = {}
        for x in isupply:
            s[x] = (costs[x][g[x][1]] - costs[x][g[x][0]]) if len(g[x]) > 1 else costs[x][g[x][0]]

        f = max(d, key=lambda n: d[n])
        t = max(s, key=lambda n: s[n])

        t, f = (f, g[f][0]) if d[f] > s[t] else (g[t][0], t)

        v = min(isupply[f], idemand[t])
        res[f][t] += v
        idemand[t] -= v

        if idemand[t] == 0:
            for k, n in isupply.items():
                if n != 0:
                    g[k].remove(t)
            del g[t]
            del idemand[t]
        isupply[f] -= v
        if isupply[f] == 0:
            for k, n in idemand.items():
                if n != 0:
                    g[k].remove(f)
            del g[f]
            del isupply[f]

    ans = []
    cost = 0
    for g in sorted(costs):
        temp = []
        for n in cols:
            y = res[g][n]
            temp.append(y)
            cost += y * costs[g][n]
        ans.append(temp)

    fans = []
    for i in range(len(supply)):
        for j in range(len(demand)):
            temp = (i, j)
            temp2 = []
            if ans[i][j] != 0:
                temp2.append(temp)
                temp2.append(ans[i][j])
                fans.append(temp2)

    return fans



def get_ws(bfs, costs, us, vs):
    ws = []
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            non_basic = all([p[0] != i or p[1] != j for p, v in bfs])
            if non_basic:
                ws.append(((i, j), us[i] + vs[j] - cost))

    return ws

def can_be_improved(ws):
    for p, v in ws:
        if v > 0: return True
    return False

def get_entering_variable_position(ws):
    ws_copy = ws.copy()
    ws_copy.sort(key=lambda w: w[1])
    return ws_copy[-1][0]

def get_possible_next_nodes(loop, not_visited):
    last_node = loop[-1]
    nodes_in_row = [n for n in not_visited if n[0] == last_node[0]]
    nodes_in_column = [n for n in not_visited if n[1] == last_node[1]]
    if len(loop) < 2:
        return nodes_in_row + nodes_in_column
    else:
        prev_node = loop[-2]
        row_move = prev_node[0] == last_node[0]
        if row_move: return nodes_in_column
        return nodes_in_row


def get_loop(bv_positions, ev_position):
    def inner(loop):
        if len(loop) > 3:
            can_be_closed = len(get_possible_next_nodes(loop, [ev_position])) == 1
            if can_be_closed: return loop

        not_visited = list(set(bv_positions) - set(loop))
        possible_next_nodes = get_possible_next_nodes(loop, not_visited)
        for next_node in possible_next_nodes:
            new_loop = inner(loop + [next_node])
            if new_loop: return new_loop

    return inner([ev_position])


def loop_pivoting(bfs, loop):
    even_cells = loop[0::2]
    odd_cells = loop[1::2]
    get_bv = lambda pos: next(v for p, v in bfs if p == pos)
    leaving_position = sorted(odd_cells, key=get_bv)[0]
    leaving_value = get_bv(leaving_position)

    new_bfs = []
    for p, v in [bv for bv in bfs if bv[0] != leaving_position] + [(loop[0], 0)]:
        if p in even_cells:
            v += leaving_value
        elif p in odd_cells:
            v -= leaving_value
        new_bfs.append((p, v))

    return new_bfs

def get_balanced_tp(supply, demand, costs, penalties=None):
    total_supply = sum(supply)
    total_demand = sum(demand)

    if total_supply < total_demand:
        if penalties is None:
            raise Exception('Supply less than demand, penalties required')
        new_supply = supply + [total_demand - total_supply]
        new_costs = costs + [penalties]
        return new_supply, demand, new_costs
    if total_supply > total_demand:
        new_demand = demand + [total_supply - total_demand]
        new_costs = costs + [[0 for _ in demand]]
        return supply, new_demand, new_costs
    return supply, demand, costs


def transportation_simplex_method(supply, demand, costs, penalties=None):
    balanced_supply, balanced_demand, balanced_costs = get_balanced_tp(
        supply, demand, costs
    )

    def inner(bfs):
        us, vs = get_us_and_vs(bfs, balanced_costs)
        ws = get_ws(bfs, balanced_costs, us, vs)
        if can_be_improved(ws):
            ev_position = get_entering_variable_position(ws)
            loop = get_loop([p for p, v in bfs], ev_position)
            return inner(loop_pivoting(bfs, loop))
        return bfs

    basic_variables = inner(vam(balanced_supply, balanced_demand))
    solution = np.zeros((len(costs), len(costs[0])))
    for (i, j), v in basic_variables:
        solution[i][j] = v

    return solution


def get_total_cost(costs, solution):
    total_cost = 0
    for i, row in enumerate(costs):
        for j, cost in enumerate(row):
            total_cost += cost * solution[i][j]
    return total_cost


costs = [
    [ 2, 2, 2, 1],
    [10, 8, 5, 4],
    [ 7, 6, 6, 8]
]
supply = [30, 70, 50]
demand = [40, 30, 40, 40]
bfs = vam(supply, demand)
us, vs = get_us_and_vs(bfs, costs)
ws = get_ws(bfs, costs, us, vs)
can_be_improved(ws)

ev_position = get_entering_variable_position(ws)
loop = get_loop([p for p, v in bfs], ev_position)
new_bfs = loop_pivoting(bfs, loop)

solution = transportation_simplex_method(supply, demand, costs)
print(solution)
print('total cost: ', get_total_cost(costs, solution))