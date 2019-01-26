from coopr.pyomo import *
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
from random import uniform
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ast
import pprint
from numpy import ones, vstack, arange
from numpy.linalg import lstsq
from statistics import mean
import datetime


class TSPModel:

    def __init__(self, entrada):
        self.coordinates = entrada['coords']
        self.tempo_atendimento = entrada['tempo_atendimento']
        self.tempo_km = entrada['tempo_km']
        self.horas_dia = entrada['horas_dia']
        if self.coordinates is not None:
            self.n = len(self.coordinates)
            self.all_nodes = list(self.coordinates.keys())
            self.all_nodes.sort()
            self.client_nodes = self.all_nodes[1:]
        else:
            self.n = 0
        self.d = None
        self.distance_set()
        self.start_point = None
        self.output = {
            'total_distance': 0,
            'total_time_in_hours': 0,
            'average_working_time': 0,
            'total_number_of_days': 0,
            'sequences': [],
            'clients': [],
            'salesman': None,
            'distant_clients': []
        }
        self.salesmen = None
        self.spreadsheet_name = None
        self.data = None
        self.nodes_set_f = None
        self.nodes_set_t = None
        self.nodes_set = None
        self.x = None
        self.arcs_sequence = None

    @staticmethod
    def connect_to_google():
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
            return gspread.authorize(credentials)
        except:
            print('Erro na conexcao')
            exit()

    def get_salesmen(self):
        gc = self.connect_to_google()
        wks = gc.open('vendedores ativos').get_worksheet(0)
        data = wks.get_all_records()
        self.salesmen = [{'nome': d['VENDEDOR'],
                          'id': d['COD_VENDEDOR'],
                          'origem': (float(d['Latitude']), float(d['Longitude']))} for d in data]

    def get_coordinates(self, salesman_list_position):
        gc = self.connect_to_google()
        wks = gc.open('lista de clientes atualizada').worksheet('sheet1')
        self.data = wks.get_all_records()
        self.coordinates = {}
        self.coordinates[0] = self.salesmen[salesman_list_position]['origem']
        i = 1
        for line in self.data:
            if line['VENDEDOR'] == self.salesmen[salesman_list_position]['id']:
                self.coordinates[i] = (float(line['Latitude']), float(line['Longitude']))
                self.output['clients'].append(
                    {'id': i, 'entidade_id': line['CLIENTEID'], 'description': line['NOMECLIENTE']})
                i += 1
        self.output['salesman'] = {'id': 0,
                                   'vendedor_id': self.salesmen[salesman_list_position]['id'],
                                   'description': self.salesmen[salesman_list_position]['nome']}

        self.all_nodes = list(self.coordinates.keys())
        self.all_nodes.sort()
        self.client_nodes = self.all_nodes[1:]

    def check_all(self):
        for salesman_list_position in range(len(self.salesmen)):
            # print('-' * 50)
            self.get_coordinates(salesman_list_position)
            self.distance_set()
            self.check_data(salesman_list_position, complete=False, verbose=False)

    def check_data(self, salesman_list_position, complete=True, verbose=True):
        msg_showed = False
        out_coordinates_ids = []
        for j in self.client_nodes:
            if self.d[0, j] * self.tempo_km * 2 + self.tempo_atendimento > self.horas_dia:
                if not msg_showed:
                    print('Vendedor {}:'.format(self.output['salesman']['vendedor_id']))
                    print('As coordenadas abaixo estão muito distantes da origem:')
                msg_showed = True
                print('-- cliente {:>2} - {:>2} - {:>2} horas da origem'.format(
                    self.output['clients'][j - 1]['entidade_id'],
                    self.output['clients'][j - 1]['description'],
                    round(self.d[0, j] * self.tempo_km + self.tempo_atendimento, 2)))
                out_coordinates_ids.append(j)
        if verbose:
            if msg_showed and complete:
                print('Coordenadas removidas => ', out_coordinates_ids)
                self.coordinates = {}
                self.coordinates[0] = self.salesmen[salesman_list_position]['origem']
                i = 1
                for line in self.data:
                    if line['VENDEDOR'] == self.salesmen[salesman_list_position]['id']:
                        if i not in out_coordinates_ids:
                            self.coordinates[i] = (float(line['Latitude']), float(line['Longitude']))
                            self.output['clients'].append(
                                {'id': i, 'entidade_id': line['CLIENTEID'], 'description': line['NOMECLIENTE']})
                        else:
                            self.output['distant_clients'].append(
                                {'id': i, 'entidade_id': line['CLIENTEID'], 'description': line['NOMECLIENTE']})
                        i += 1
                self.all_nodes = list(self.coordinates.keys())
                self.all_nodes.sort()
                self.client_nodes = self.all_nodes[1:]
            elif not msg_showed:
                print('Vendedor {}:'.format(self.output['salesman']['vendedor_id']))
                print('Sem coordenadas distantes!')
                return False
            if msg_showed:
                return True

    def random_sample(self, size, latitude_range=(-3.90, -3.47), longitude_range=(-39.31, -38.18)):
        self.coordinates = {
            i: (uniform(latitude_range[0], latitude_range[1]), uniform(longitude_range[0], longitude_range[1]))
            for i in range(size)
        }
        self.n = len(self.coordinates)
        self.all_nodes = list(self.coordinates.keys())
        self.all_nodes.sort()
        self.client_nodes = self.all_nodes[1:]
        self.distance_set()

    def distance_set(self):
        earth_radius = 6373.0
        self.d = dict()
        for i in self.all_nodes:
            for j in self.all_nodes:
                lat1 = radians(self.coordinates[i][0])
                lon1 = radians(self.coordinates[i][1])
                lat2 = radians(self.coordinates[j][0])
                lon2 = radians(self.coordinates[j][1])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                self.d[i, j] = earth_radius * c

    def plot_coordinates(self):
        x = [self.coordinates[i][0] for i in self.client_nodes]
        y = [self.coordinates[i][1] for i in self.client_nodes]
        center_x = [center[0] for center in self.c]
        center_y = [center[1] for center in self.c]
        fig, ax = plt.subplots()
        ax.plot(self.coordinates[0][0], self.coordinates[0][1], 'o')
        ax.plot(x, y, 'o')
        ax.plot(center_x, center_y, 'o')
        ax.plot()
        plt.show()

    @staticmethod
    def distance(x, c):
        return sqrt((x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)

    def k_mean_cluster(self, k):
        # coordinates = list(self.coordinates.values())
        weight = len(self.coordinates) // 5
        max_size = len(self.coordinates) // k + 1
        error = float('inf')
        min_x = min(coord[0] for coord in self.coordinates.values())
        max_x = max(coord[0] for coord in self.coordinates.values())
        min_y = min(coord[1] for coord in self.coordinates.values())
        max_y = max(coord[1] for coord in self.coordinates.values())
        self.c = [(uniform(min_x, max_x), uniform(min_y, max_y)) for i in range(k)]
        for j in range(100):
            self.nodes_set_c = [[0] for i in range(k)]
            for node, x in zip(self.coordinates.keys(), self.coordinates.values()):
                if node != 0:
                    min_distance = float('inf')
                    c_id = None
                    for i, center in enumerate(self.c):
                        if len(self.nodes_set_c[i]) <= max_size:
                            if self.distance(x, center) < min_distance:
                                min_distance = self.distance(x, center)
                                c_id = i
                    self.nodes_set_c[c_id].append(node)
            new_c = [(mean([self.coordinates[node][0] for node in nodes] + [self.coordinates[0][0]]),
                      mean([self.coordinates[node][1] for node in nodes] + [self.coordinates[0][1]])) for nodes in self.nodes_set_c]
            # print(new_c)
            error = sum(self.distance(center, new_center) for center, new_center in zip(self.c, new_c))
            self.c = list(new_c)
            if error < 0.01:
                break
        print('cluster set:----n={}'.format(len(self.nodes_set_c)))
        print(self.nodes_set_c)

    def k_mean_cluster2(self, k, max_size=20):
        # coordinates = list(self.coordinates.values())
        min_x = min(coord[0] for coord in self.coordinates.values())
        max_x = max(coord[0] for coord in self.coordinates.values())
        min_y = min(coord[1] for coord in self.coordinates.values())
        max_y = max(coord[1] for coord in self.coordinates.values())
        c = [(uniform(min_x, max_x), uniform(min_y, max_y)) for i in range(k)]
        error = float('inf')
        while error >= 0.01:
            self.nodes_set_c = [[0] for i in range(k)]
            for node, x in zip(self.coordinates.keys(), self.coordinates.values()):
                if node != 0:
                    min_distance = float('inf')
                    c_id = None
                    for i, center in enumerate(c):
                        if self.distance(x, center) < min_distance:
                            min_distance = self.distance(x, center)
                            c_id = i
                    self.nodes_set_c[c_id].append(node)
            new_c = [(mean([self.coordinates[node][0] for node in nodes] + [self.coordinates[0][0]]),
                      mean([self.coordinates[node][1] for node in nodes]  + [self.coordinates[0][1]])) for nodes in self.nodes_set_c]
            # print(new_c)
            error = sum(self.distance(center, new_center) for center, new_center in zip(c, new_c))
            c = list(new_c)
        repeat = True
        print(self.nodes_set_c)
        while repeat:
            over_size_cluster = False
            for id, cluster in enumerate(self.nodes_set_c):
                if len(cluster) > max_size:
                    over_size_cluster = True
                    min_x = min(coord[0] for i, coord in enumerate(self.coordinates.values()) if i in cluster)
                    max_x = max(coord[0] for i, coord in enumerate(self.coordinates.values()) if i in cluster)
                    min_y = min(coord[1] for i, coord in enumerate(self.coordinates.values()) if i in cluster)
                    max_y = max(coord[1] for i, coord in enumerate(self.coordinates.values()) if i in cluster)
                    c = [(uniform(min_x, max_x), uniform(min_y, max_y)) for i in range(2)]
                    error = float('inf')
                    while error >= 0.01:
                        temp_nodes_set_c = [[0] for i in range(2)]
                        for node, x in zip(self.coordinates.keys(), self.coordinates.values()):
                            if node != 0 and node in cluster:
                                min_distance = float('inf')
                                c_id = None
                                for i, center in enumerate(c):
                                    if self.distance(x, center) < min_distance:
                                        min_distance = self.distance(x, center)
                                        c_id = i
                                temp_nodes_set_c[c_id].append(node)
                        new_c = [(mean([self.coordinates[node][0] for node in nodes] + [self.coordinates[0][0]]),
                                  mean([self.coordinates[node][1] for node in nodes] + [self.coordinates[0][1]])) for
                                 nodes in temp_nodes_set_c]
                        # print(new_c)
                        error = sum(self.distance(center, new_center) for center, new_center in zip(c, new_c))
                        c = list(new_c)
                    self.nodes_set_c.pop(id)
                    self.nodes_set_c += temp_nodes_set_c
                    break
            if not over_size_cluster:
                repeat = False
        temp = []
        for nodes in self.nodes_set_c:
            if len(nodes) > 1:
                temp.append(nodes)
        self.nodes_set_c = list(temp)

        print('cluster set:----n={}'.format(len(self.nodes_set_c)))
        print(self.nodes_set_c)

    def solve_partitioned(self, timelimit, partition, k=4):
        if partition == 'cluster':
            self.k_mean_cluster(k)
            nodes_set = self.nodes_set_c
        elif partition == 1:
            nodes_set = [self.all_nodes]
        else:
            print('Modelo de particao --{}-- não existe'.format(partition))
            exit()
        self.x = {(i, j): 0 for i in self.all_nodes for j in self.all_nodes}
        for nodes in nodes_set:
            print('solving for: ', nodes)
            print('with {} nodes'.format(len(nodes)))
            n = len(nodes)
            model = ConcreteModel()
            # parâmetros
            model.d = self.d
            model.all_nodes = Set(
                initialize=range(len(nodes))
            )
            model.client_nodes = Set(
                initialize=range(1, len(nodes))
            )
            model.days = Set(
                initialize=list(range(len(nodes) // 2))
            )
            # variáveis
            model.x = Var(model.all_nodes, model.all_nodes, model.days, within=Binary)
            model.z = Var(model.all_nodes, model.days, within=Binary)
            model.u = Var(model.all_nodes, within=NonNegativeIntegers)
            model.m = Var(within=NonNegativeIntegers)
            print('variables created')
            model.obj = Objective(
                expr=sum(
                    model.d[nodes[i], nodes[j]] * model.x[i, j, k] for i in model.all_nodes for j in model.all_nodes for k in model.days),
                sense=minimize
            )
            model.constset4 = ConstraintList()
            for k in model.days:
                model.constset4.add(
                    sum((self.tempo_km * self.d[nodes[i], nodes[j]] + self.tempo_atendimento) * model.x[i, j, k]
                        for i in model.all_nodes
                        for j in model.client_nodes) +
                    sum(self.tempo_km * self.d[nodes[i], nodes[0]] * model.x[i, 0, k] for i in
                        model.client_nodes) <= self.horas_dia
                )
            # restrições de saída e chegada do ponto inicial
            model.const1 = Constraint(
                expr=sum(model.x[0, j, k]
                         for j in model.client_nodes
                         for k in model.days) == model.m
            )
            model.const2 = Constraint(
                expr=sum(model.x[i, 0, k]
                         for i in model.client_nodes
                         for k in model.days) == model.m
            )
            # restrições de chegada e saída dos pontos de visita
            model.constset1 = ConstraintList()
            for j in model.client_nodes:
                model.constset1.add(
                    sum(model.x[i, j, k]
                        for i in model.all_nodes for k in model.days if i != j) == 1
                )
            model.constset2 = ConstraintList()
            for i in model.client_nodes:
                model.constset2.add(
                    sum(model.x[i, j, k] for j in model.all_nodes for k in model.days if i != j) == 1
                )
            # restrição de eliminação de subrotas (Miller-Tucker-Zemlin)
            model.constset3 = ConstraintList()
            for i in model.client_nodes:
                for j in model.client_nodes:
                    for k in model.days:
                        if i != j:
                            model.constset3.add(
                                model.u[i] - model.u[j] + (n - 1) * model.x[i, j, k] <= n - 2
                            )

            model.constset5 = ConstraintList()
            for i in model.all_nodes:
                for j in model.all_nodes:
                    for k in model.days:
                        if i != j:
                            model.constset5.add(
                                model.x[i, j, k] <= model.z[i, k]
                            )
                            model.constset5.add(
                                model.x[i, j, k] <= model.z[j, k]
                            )
            model.constset6 = ConstraintList()
            for j in model.client_nodes:
                model.constset6.add(
                    sum(model.z[j, k] for k in model.days) == 1
                )

            # solver
            print('model constructed')
            solver = SolverFactory('cplex')
            print('started at {}'.format(datetime.datetime.now()))
            solver.options['timelimit'] = timelimit
            solver.options['emphasis_memory'] = 'y'
            solver.solve(model, tee=False)
            print('finished')
            for i in model.all_nodes:
                for j in model.all_nodes:
                    for k in model.days:
                        try:
                            if model.x[i, j, k].value is not None:
                                self.output['total_distance'] += self.d[i, j] * model.x[i, j, k].value
                        except:
                            pass
            for i in model.all_nodes:
                for j in model.all_nodes:
                    if sum(model.x[i, j, k].value for k in model.days if model.x[i, j, k].value is not None) == 1.0:
                        self.x[nodes[i], nodes[j]] = 1
                    else:
                        self.x[nodes[i], nodes[j]] = 0

    def get_results(self):
        arcs = []
        for i in self.all_nodes:
            for j in self.all_nodes:
                if self.x[i, j] == 1:
                    arcs.append((i, j))
        self.arcs_sequence = []
        for r in arcs:
            if r[0] == 0:
                self.arcs_sequence.append([r])
        for k in range(len(self.arcs_sequence)):
            finish = False
            while not finish:
                for r in arcs:
                    if self.arcs_sequence[k][-1][1] == r[0]:
                        self.arcs_sequence[k].append(r)
                        if r[1] == 0:
                            finish = True

    def generate_results(self, i):
        self.get_results()
        dia = 1
        for arcs in self.arcs_sequence:
            self.output['sequences'].append({'day': dia, 'sequence': [0], 'time(h)': 0, 'distance(km)': 0})
            for r in arcs:
                if r[1] != 0:
                    self.output['sequences'][-1]['time(h)'] += self.tempo_atendimento + self.d[r[0], r[1]] * self.tempo_km
                else:
                    self.output['sequences'][-1]['time(h)'] += self.d[r[0], r[1]] * self.tempo_km
                self.output['sequences'][-1]['sequence'].append(r[1])
                self.output['sequences'][-1]['distance(km)'] += self.d[r[0], r[1]]
            dia += 1
        self.output['total_time_in_hours'] = sum(seq['time(h)'] for seq in self.output['sequences'])
        self.output['total_number_of_days'] = dia - 1
        self.output['average_working_time'] = self.output['total_time_in_hours'] / dia - 1
        file = open('results/vendedor_{}_output.txt'.format(self.salesmen[i]['id']), 'w')
        file.write(str(self.output))
        # print('Objeto de resposta do modelo:')
        # print(self.output)

    def show_results(self):
        dia = 1
        for arcs in self.arcs_sequence:
            time_span = 0
            distance = 0
            resp = '0 --> '
            for r in arcs:
                if r[1] != 0:
                    resp += ' {} --> '.format(r[1])
                    time_span += self.tempo_atendimento + self.d[r[0], r[1]] * self.tempo_km
                else:
                    resp += ' {}'.format(r[1])
                    time_span += self.d[r[0], r[1]] * self.tempo_km
                distance += self.d[r[0], r[1]]
            print('dia {}: {} | tempo (h): {} | dist (km): {}'.format(dia, resp, time_span, distance))
            dia += 1

    def plot_solution(self, i):
        from itertools import cycle
        cycol = cycle('bgrcmk')
        x = [self.coordinates[i][0] for i in self.client_nodes]
        y = [self.coordinates[i][1] for i in self.client_nodes]
        fig, ax = plt.subplots()
        ax.plot(self.coordinates[0][0], self.coordinates[0][1], 'o')
        ax.plot(x, y, 'o')
        x = [self.coordinates[i][0] for i in self.all_nodes]
        y = [self.coordinates[i][1] for i in self.all_nodes]
        for arcs in self.arcs_sequence:
            color = next(cycol)
            for arc in arcs:
                try:
                    self.connect_points(x, y, arc[0], arc[1], color)
                except:
                    pass
        plt.savefig('results/vendedor_{}_output.png'.format(self.salesmen[i]['id']))

    def save_on_google(self, i):
        gc = self.connect_to_google()
        wks = gc.open('resultados - rotas').worksheet(self.salesmen[i]['id'])
        i = 2
        columns = ['ENTIDADEID', 'DESCRICAO', 'ENDEREÇO', 'ROTAID', 'SEQUÊNCIA']
        for day in self.output['sequences']:
            j = 1
            for node in day['sequence']:
                if node != 0:
                    cell_list = wks.range(i, 1, i, 5)
                    exp_index = next((index for (index, d) in enumerate(self.data) if d["id"] == node), None)
                    for cell, column in zip(cell_list, columns):
                        if column != 'dia' and column != 'ordem':
                            cell.value = self.data[exp_index][column]
                        elif column == 'dia':
                            cell.value = day['day']
                        elif column == 'ordem':
                            cell.value = j
                    j += 1
                    wks.update_cells(cell_list)
                    i += 1
        self.save_abstract()

    def save_abstract(self):
        gc = self.connect_to_google()
        wks = gc.open(self.spreadsheet_name).get_worksheet(2)
        wks.update_cell(2, 2, self.output['total_number_of_days'])
        wks.update_cell(3, 2, self.output['total_distance'])
        wks.update_cell(4, 2, self.output['total_time_in_hours'])
        wks.update_cell(5, 2, self.output['average_working_time'])

    def save_on_google_from_file(self, spreadsheet='CARTEIRA DE CLIENTES AREA 18  JULHO 18', filename='backup_output.tx'):
        file = open(filename, 'r')
        self.output = eval(file.read())
        print(self.output)
        exit()
        self.save_on_google()

    @staticmethod
    def connect_points(x, y, p1, p2, color):
        x1, x2 = x[p1], x[p2]
        y1, y2 = y[p1], y[p2]
        plt.plot([x1, x2], [y1, y2], c=color)


input_data = {
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
    'tempo_atendimento': 40 / 60,
    'tempo_km': 1 / 30,
    'horas_dia': 8
}

if __name__ == '__main__':
    # instance.start_point = (-3.7897703, -38.6155416)
    # instance.spreadsheet_name = 'CARTEIRA DE CLIENTES AREA 18  JULHO 18'
    # instance.get_data()
    # instance.solve_partitioned(timelimit=60, partition='cluster')
    # instance.generate_results()
    # instance.save_on_google()
    # instance.plot_solution()
    instance1 = TSPModel(input_data)
    instance1.get_salesmen()
    # instance1.check_all()
    # exit()
    t = 600
    # len(instance1.salesman)
    # sales_id = [7162, 7307]
    #             4487, 4180, 3840, 3446, 3331, 3297, 2413, 2307, 2173, 2172, 1266]
    for i in range(len(instance1.salesmen)):
        instance = TSPModel(input_data)
        # try:
        instance.salesmen = instance1.salesmen
        print('--------------------------', instance.salesmen[i]['id'])
        instance.get_coordinates(i)
        instance.distance_set()
        too_distante = instance.check_data(i)
        if len(instance.d) > 1:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            # instance.distance_set()
            n = len(instance.coordinates) // 26 + 1
            if n == 1:
                instance.solve_partitioned(timelimit=t, partition=1)
            else:
                instance.solve_partitioned(timelimit=t, partition='cluster', k=n)
            instance.generate_results(i)
            instance.plot_solution(i)
        exit()
        # except:
        #     pass
    # instance.solve_partitioned(timelimit=600, partition=1)
    # instance.generate_results()
    # instance.k_mean_cluster(4)
    # instance.plot_coordinates()
