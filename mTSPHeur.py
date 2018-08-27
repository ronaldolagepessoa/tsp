from math import sin, cos, sqrt, atan2, radians
from pprint import pprint
from random import random, shuffle
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ast


scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']

try:
    credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
    gc = gspread.authorize(credentials)
except:
    pass


def get_data(start_point):
    wks = gc.open('CARTEIRA DE CLIENTES AREA 18  JULHO 18').get_worksheet(0)
    data = wks.get_all_records()
    coords = {i + 1: ast.literal_eval(coord['Latitude/Longitude']) for i, coord in enumerate(data)}
    coords[0] = start_point
    return coords


def distance_set(coords):
    R = 6373.0
    d = dict()
    n = len(coords)
    for i in range(n):
        for j in range(n):
            lat1 = radians(coords[i][0])
            lon1 = radians(coords[i][1])
            lat2 = radians(coords[j][0])
            lon2 = radians(coords[j][1])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            d[i, j] = R * c
    return d


def fitness(solution, entrada):
    i = 0
    total_distance = 0
    while i < len(solution):
        need_to_return = False
        total_day_time = d[0, solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
        total_distance += d[0, solution[i]]
        if total_day_time + d[solution[i], 0] * entrada['tempo_km'] >= entrada['horas_dia']:
            total_day_time += d[solution[i], 0] * entrada['tempo_km']
            total_distance += d[solution[i], 0]
            i += 1
            need_to_return = True
            if i == len(solution):
                return total_distance
        else:
            i += 1
        if i == len(solution) - 1:
            total_day_time += d[solution[i - 1], solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
            total_distance += d[solution[i - 1], solution[i]]
            total_day_time += d[solution[i], 0] * entrada['tempo_km']
            total_distance += d[solution[i], 0]
            return total_distance
        while not need_to_return:
            total_day_time += d[solution[i - 1], solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
            total_distance += d[solution[i - 1], solution[i]]
            if total_day_time + d[solution[i], 0] * entrada['tempo_km'] >= entrada['horas_dia']:
                total_day_time += d[solution[i], 0] * entrada['tempo_km']
                total_distance += d[solution[i], 0]
                need_to_return = True
            else:
                need_to_return = False
            i += 1
            if i == len(solution) - 1:
                total_day_time = d[solution[i - 1], solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
                total_distance += d[solution[i - 1], solution[i]]
                total_day_time += d[solution[i], 0] * entrada['tempo_km']
                total_distance += d[solution[i], 0]
                return total_distance


def result(solution, entrada):
    i = 0
    path = [0]
    while i < len(solution):
        need_to_return = False
        total_day_time = d[0, solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
        path.append(solution[i])
        if total_day_time + d[solution[i], 0] * entrada['tempo_km'] >= entrada['horas_dia']:
            total_day_time += d[solution[i], 0] * entrada['tempo_km']
            path.append(0)
            i += 1
            need_to_return = True
            if i == len(solution):
                return path
        else:
            i += 1
        if i == len(solution) - 1:
            total_day_time += d[solution[i - 1], solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
            path.append(solution[i])
            total_day_time += d[solution[i], 0] * entrada['tempo_km']
            path.append(solution[i])
            path.append(0)
            return path
        while not need_to_return:
            total_day_time += d[solution[i - 1], solution[i]] * entrada['tempo_km'] + entrada['tempo_atendimento']
            path.append(solution[i])
            if total_day_time + d[solution[i], 0] * entrada['tempo_km'] >= entrada['horas_dia']:
                total_day_time += d[solution[i], 0] * entrada['tempo_km']
                path.append(0)
                need_to_return = True
            else:
                need_to_return = False
            i += 1
            if i == len(solution) - 1:
                total_day_time += d[solution[i], 0] * entrada['tempo_km']
                path.append(solution[i])
                path.append(0)
                return path

def local_search(solution, entrada, opt=2):
    best_cost = fitness(solution, entrada)
    best_solution = list(solution)
    count = 0
    r = list(range(len(solution)))
    shuffle(r)
    for i in range(len(solution) - 1):
        for j in range(i + 1, len(solution)):
            solution[r[i]], solution[r[j]] = solution[r[j]], solution[r[i]]
            cur_cost = fitness(solution, entrada)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_solution = list(solution)
                count += 1
            if count == 2:
                return best_cost, best_solution
    return best_cost, best_solution


def greedy_adaptative(entrada):
    nodes = list(range(1, len(entrada['coords'])))
    d = distance_set(entrada['coords'])
    i = 0
    solution = []
    while len(nodes) > 0:
        inverted = [1 / d[i, j] for j in nodes]
        total = sum(inverted)
        prob = [inv / total for inv in inverted]
        solution.append(random_pick(nodes, prob))
        nodes.remove(solution[-1])
    return solution


def random_pick(some_list, probabilities):
    x = random()
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


def GRASP(entrada, n=1000):
    best_solution = list(range(1, len(entrada['coords'])))
    best_cost = fitness(best_solution, entrada)
    for i in range(n):
        if i % 100 == 0:
            print('ite {}'.format(i))
        solution = greedy_adaptative(entrada)
        cur_cost, cur_solution = local_search(solution, entrada)
        if cur_cost < best_cost:
            best_cost = cur_cost
            best_solution = list(cur_solution)
            print(best_cost, '{}%'.format(i / n * 100))
            print(result(best_solution, entrada))
    return best_cost, best_solution


def report(path):
    output = []
    for i in range(len(path) - 1):
        if path[i] == 0:
            output.append([])
        else:
            output[-1].append(path[i])
    return output


def plot_solution(entrada, ouput):
    from itertools import cycle
    n = len(entrada['coords'])
    cycol = cycle('bgrcmk')
    x = [entrada['coords'][i][0] for i in range(1, n)]
    y = [entrada['coords'][i][1] for i in range(1, n)]
    fig, ax = plt.subplots()
    ax.plot(entrada['coords'][0][0], entrada['coords'][0][1], 'o')
    ax.plot(x, y, 'o')
    x = [entrada['coords'][i][0] for i in range(n)]
    y = [entrada['coords'][i][1] for i in range(n)]
    for nodes in ouput:
        color = next(cycol)
        temp = [0] + nodes + [0]
        for k in range(len(temp) - 1):
            connectpoints(x, y, temp[k], temp[k + 1], color)
    plt.show()


def connectpoints(x, y, p1, p2, color):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    plt.plot([x1, x2], [y1, y2], c=color)



entrada = {
    'coords': {
        0: (-3.8412925646756273, -38.19748083389053),
        1: (-3.8076093143793464, -38.31531507188827),
        2: (-3.807076473247719, -38.534964161996015),
        3: (-3.8937784109629603, -38.276942506062476),
        4: (-3.4984004291098874, -38.20459829781128),
        5: (-3.624210043291873, -39.228378700101416),
        6: (-3.547778842055511, -38.40887494561352),
        7: (-3.660961612846057, -38.64362570444013),
        8: (-3.8977471504396077, -38.429476020171556),
        9: (-3.67754845892978, -39.014612378325545),
        10: (-3.730734725846105, -38.61605187627415),
        11: (-3.843487789860545, -38.74566546750679),
        12: (-3.8627966544258188, -38.587290555785295),
        13: (-3.553873034380957, -38.819609163825476)},
    'tempo_atendimento': 0.4,
    'tempo_km': 1 / 40,
    'horas_dia': 8
}

entrada['coords'] = get_data((-3.7897703, -38.6155416))
print('ok')

if __name__ == '__main__':
    d = distance_set(entrada['coords'])
    print('distance_set = ok')
    best_cost, best_solution = GRASP(entrada, n=30000)
    path = result(best_solution, entrada)
    ouput = report(path)
    plot_solution(entrada, ouput)



