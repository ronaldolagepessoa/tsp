from coopr.pyomo import *
from math import sin, cos, sqrt, atan2, radians
import matplotlib.pyplot as plt
from random import uniform
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ast
from numpy import ones, vstack, arange
from numpy.linalg import lstsq


class TSPModel:

    def __init__(self, entrada):
        self.coords = entrada['coords']
        self.tempo_atendimento = entrada['tempo_atendimento']
        self.tempo_km = entrada['tempo_km']
        self.horas_dia = entrada['horas_dia']
        if self.coords != None:
            self.n = len(self.coords)
            self.all_nodes = list(self.coords.keys())
            self.all_nodes.sort()
            self.client_nodes = self.all_nodes[1:]
        else:
            self.n = 0
        self.distance_set()
        self.start_point = None
        self.output = {
            'total_distance': 0,
            'total_time_in_hours': 0,
            'average_working_time': 0,
            'total_number_of_days': 0,
            'sequences': []
        }
        self.spreadsheet_name = None

    def get_data(self):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
            gc = gspread.authorize(credentials)
        except:
            print('Erro na conexcao')
            exit()
        wks = gc.open('CARTEIRA DE CLIENTES AREA 18  JULHO 18').get_worksheet(0)
        self.data = wks.get_all_records()
        if self.start_point != None:
            self.coords = {i + 1: ast.literal_eval(coord['Latitude/Longitude']) for i, coord in enumerate(self.data)}
            self.coords[0] = self.start_point
            self.all_nodes = list(self.coords.keys())
            self.all_nodes.sort()
            self.client_nodes = self.all_nodes[1:]
        else:
            self.coords = {i: ast.literal_eval(coord['Latitude/Longitude']) for i, coord in enumerate(self.data)}
            self.all_nodes = list(self.coords.keys())
            self.all_nodes.sort()
            self.client_nodes = self.all_nodes[1:]
        self.n = len(self.coords)
        self.distance_set()
        msg_showed = False
        for j in self.all_nodes:
            if self.d[0, j] * self.tempo_km * 2 + self.tempo_atendimento > self.horas_dia:
                if not msg_showed:
                    print('As coordenadas abaixo estão muito distantes da origem:')
                msg_showed = True
                print('t[{}, {}] = {}'.format(0, j, self.d[0, j] * self.tempo_km + self.tempo_atendimento))
        if msg_showed:
            exit()

    def random_sample(self, size, latitude_range=(-3.90, -3.47), longitude_range=(-39.31, -38.18)):
        self.coords = {
            i: (uniform(latitude_range[0], latitude_range[1]), uniform(longitude_range[0], longitude_range[1]))
            for i in range(size)
        }
        self.n = len(self.coords)
        self.all_nodes = list(self.coords.keys())
        self.all_nodes.sort()
        self.client_nodes = self.all_nodes[1:]
        self.distance_set()

    def distance_set(self):
        R = 6373.0
        self.d = dict()
        for i in self.all_nodes:
            for j in self.all_nodes:
                lat1 = radians(self.coords[i][0])
                lon1 = radians(self.coords[i][1])
                lat2 = radians(self.coords[j][0])
                lon2 = radians(self.coords[j][1])
                dlon = lon2 - lon1
                dlat = lat2 - lat1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                self.d[i, j] = R * c

    def plot_coords(self):
        x = [self.coords[i][0] for i in self.client_nodes]
        y = [self.coords[i][1] for i in self.client_nodes]
        fig, ax = plt.subplots()
        ax.plot(self.coords[0][0], self.coords[0][1], 'o')
        ax.plot(x, y, 'o')
        plt.show()

    def rotate(self, x, y, cx, cy, ang):
        theta = radians(ang)
        rx = ((x - cx) * cos(theta) + (y - cy) * sin(theta)) + cx
        ry = (-(x - cx) * sin(theta) + (y - cy) * cos(theta)) + cy
        return rx, ry

    def linear_partition(self):
        cx, cy = self.coords[0][0], self.coords[0][1]
        self.nodes_set = [[0], [0]]
        for ang in arange(0, 180, 0.2):
            x, y = self.rotate(cx, cy * 1.5, cx, cy, ang)
            points = [(cx, cy), (x, y)]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords, rcond=None)[0]
            count = 0
            for i in range(1, self.n):
                if self.coords[i][1] - m * self.coords[i][0] <= c:
                    count += 1
            if count <= (self.n - 1) // 2 + 1 and count >= (self.n - 1) // 2 - 1:
                for i in range(1, self.n):
                    if self.coords[i][1] - m * self.coords[i][0] < c:
                        self.nodes_set[0].append(i)
                    else:
                        self.nodes_set[1].append(i)
                return self.nodes_set

    def linear_partition_four(self):
        self.linear_partition()
        cx, cy = self.coords[0][0], self.coords[0][1]
        self.nodes_set_f = [[0], [0], [0], [0]]
        k = 0
        for j in range(2):
            finished = False
            for ang in arange(0, 180, 0.2):
                x, y = self.rotate(cx, cy * 1.5, cx, cy, ang)
                points = [(cx, cy), (x, y)]
                x_coords, y_coords = zip(*points)
                A = vstack([x_coords, ones(len(x_coords))]).T
                m, c = lstsq(A, y_coords, rcond=None)[0]
                count = 0
                for i in self.nodes_set[j]:
                    if self.coords[i][1] - m * self.coords[i][0] <= c and i != 0:
                        count += 1
                if count <= (len(self.nodes_set[j]) - 1) // 2 + 1 and count >= (len(self.nodes_set[j]) - 1) // 2 - 1:
                    finished = True
                    for i in self.nodes_set[j]:
                        if i != 0:
                            if self.coords[i][1] - m * self.coords[i][0] < c:
                                self.nodes_set_f[j + k].append(i)
                            else:
                                self.nodes_set_f[j + k + 1].append(i)
                if finished:
                    break
            k += 1
        print('Coordenadas particionadas:')
        for item in self.nodes_set_f:
            print(item)

    def linear_partition_three(self):
        cx, cy = self.coords[0][0], self.coords[0][1]
        self.nodes_set_t = [[0], [0], [0]]
        temp = []
        for ang in arange(0, 180, 0.2):
            x, y = self.rotate(cx, cy * 1.5, cx, cy, ang)
            points = [(cx, cy), (x, y)]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords, rcond=None)[0]
            count = 0
            finished = False
            for i in range(1, self.n):
                if self.coords[i][1] - m * self.coords[i][0] < c:
                    count += 1
            if count <= (self.n - 1) // 3 + 1 and count >= (self.n - 1) // 3 - 1:
                for i in range(1, self.n):
                    if self.coords[i][1] - m * self.coords[i][0] < c:
                        self.nodes_set_t[0].append(i)
                    else:
                        temp.append(i)
                finished = True
            if finished:
                break
        for ang in arange(0, 180, 0.2):
            x, y = self.rotate(cx, cy * 1.5, cx, cy, ang)
            points = [(cx, cy), (x, y)]
            x_coords, y_coords = zip(*points)
            A = vstack([x_coords, ones(len(x_coords))]).T
            m, c = lstsq(A, y_coords, rcond=None)[0]
            count = 0
            finished = False
            for i in temp:
                if self.coords[i][1] - m * self.coords[i][0] < c:
                    count += 1
            if count <= (len(temp)) // 2 + 1 and count >= (len(temp)) // 2 - 1:
                for i in temp:
                    if self.coords[i][1] - m * self.coords[i][0] < c:
                        self.nodes_set_t[1].append(i)
                    else:
                        self.nodes_set_t[2].append(i)
                finished = True
            if finished:
                break
        if not finished:
            print('no nodes generated')
            exit()
        print('Coordenadas particionadas:')
        for item in self.nodes_set_t:
            print(item)

    def pre_tsp(self):
        model = ConcreteModel()
        # parâmetros
        model.d = self.d
        model.t = self.tempo_atendimento  ## tempo padrão de atendimento em horas
        model.td = self.tempo_km  ## horas / km de distância
        model.T = self.horas_dia  ## horas de trabalho por dia

        model.all_nodes = Set(
            initialize=self.all_nodes
        )
        model.client_nodes = Set(
            initialize=self.client_nodes
        )

        # variáveis
        model.x = Var(model.all_nodes, model.all_nodes, within=Binary)
        model.u = Var(model.all_nodes, within=NonNegativeIntegers)
        # objetivo
        model.obj = Objective(
            expr=sum(
                model.d[i, j] * model.x[i, j]
                for i in model.all_nodes for j in model.all_nodes),
            sense=minimize
        )
        # restrições de chegada e saída dos pontos de visita
        model.const1 = Constraint(
            expr=sum(model.x[0, j] for j in model.client_nodes) == 2
        )
        model.const2 = Constraint(
            expr=sum(model.x[i, 0] for i in model.client_nodes) == 2
        )
        # restrições de chegada e saída dos pontos de visita
        model.constset1 = ConstraintList()
        for j in model.client_nodes:
            model.constset1.add(
                sum(model.x[i, j] for i in model.all_nodes if i != j) == 1
            )
        model.constset2 = ConstraintList()
        for i in model.client_nodes:
            model.constset2.add(
                sum(model.x[i, j] for j in model.all_nodes if i != j) == 1
            )
        # restrição de eliminação de subrotas (Miller-Tucker-Zemlin)
        model.constset3 = ConstraintList()
        for i in model.client_nodes:
            for j in model.client_nodes:
                if i != j:
                    model.constset3.add(
                        model.u[i] - model.u[j] + (self.n // 2) * model.x[i, j] <= self.n // 2 - 1
                    )
        # solver
        solver = SolverFactory('cplex')
        solver.options['timelimit'] = 120
        solver.solve(model, tee=True)
        self.m = int((model.obj.value() * model.td + self.n * model.t) / model.T) + 1
        self.p = self.n // self.m
        self.x = {(i, j): 1 if model.x[i, j].value == 1.0 else 0 for i in model.all_nodes for j in model.all_nodes}
        self.get_results()
        self.nodes_set = []
        for arcs in self.arcs_sequence:
            self.nodes_set.append([0])
            for arc in arcs:
                self.nodes_set[-1].append(arc[1])
            self.nodes_set[-1] = self.nodes_set[-1][:-1]
            self.nodes_set[-1].sort()

    def solve_full(self, timelimit):
        self.pre_tsp()
        model = ConcreteModel()
        # parâmetros
        model.m = self.m
        model.d = self.d
        model.all_nodes = Set(
            initialize=self.all_nodes
        )
        model.client_nodes = Set(
            initialize=self.client_nodes
        )
        # variáveis
        model.x = Var(model.all_nodes, model.all_nodes, within=Binary)
        model.u = Var(model.all_nodes, within=NonNegativeIntegers)
        model.obj = Objective(
            expr=sum(
                model.d[i, j] * model.x[i, j]
                for i in model.all_nodes for j in model.all_nodes),
            sense=minimize
        )
        # restrições de saída e chegada do ponto inicial
        model.const1 = Constraint(
            expr=sum(model.x[0, j] for j in model.client_nodes) == model.m
        )
        model.const2 = Constraint(
            expr=sum(model.x[i, 0] for i in model.client_nodes) == model.m
        )
        # restrições de chegada e saída dos pontos de visita
        model.constset1 = ConstraintList()
        for j in model.client_nodes:
            model.constset1.add(
                sum(model.x[i, j] for i in model.all_nodes if i != j) == 1
            )
        model.constset2 = ConstraintList()
        for i in model.client_nodes:
            model.constset2.add(
                sum(model.x[i, j] for j in model.all_nodes if i != j) == 1
            )
        # restrição de eliminação de subrotas (Miller-Tucker-Zemlin)
        model.constset3 = ConstraintList()
        for i in model.client_nodes:
            for j in model.client_nodes:
                if i != j:
                    model.constset3.add(
                        model.u[i] - model.u[j] + self.p * model.x[i, j] <= self.p - 1
                    )
        # solver
        solver = SolverFactory('cplex')
        solver.options['timelimit'] = timelimit
        solver.solve(model, tee=True)
        self.x = {(i, j): 1 if model.x[i, j].value == 1.0 else 0 for i in model.all_nodes for j in model.all_nodes}

    def solve_partitioned(self, timelimit, partition):
        if partition == 'four':
            self.linear_partition_four()
            nodes_set = self.nodes_set_f
        else:
            self.linear_partition_three()
            nodes_set = self.nodes_set_t
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
                    model.d[nodes[i], nodes[j]] * model.x[i, j, k]
                    for i in model.all_nodes for j in model.all_nodes for k in model.days),
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
            print('start')
            solver.options['timelimit'] = timelimit
            solver.options['emphasis_memory'] = 'y'
            solver.solve(model, tee=True)
            self.output['total_distance'] += sum(self.d[i, j] * model.x[i, j, k].value
                                                 for i in model.all_nodes
                                                 for j in model.all_nodes
                                                 for k in model.days if model.x[i, j, k].value != None)
            for i in model.all_nodes:
                for j in model.all_nodes:
                    if sum(model.x[i, j, k].value for k in model.days if model.x[i, j, k].value != None) == 1.0:
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

    def generate_results(self):
        self.get_results()
        dia = 1
        for arcs in self.arcs_sequence:
            time_span = 0
            distance = 0
            self.output['sequences'].append({'day': dia, 'sequence': [0], 'time(h)': 0, 'distance(km)': 0})
            for r in arcs:
                if r[1] != 0:
                    self.output['sequences'][-1]['time(h)'] += self.tempo_atendimento + self.d[r[0], r[1]] * self.tempo_km
                else:
                    self.output['sequences'][-1]['time(h)'] += self.d[r[0], r[1]] * self.tempo_km
                self.output['sequences'][-1]['sequence'].append(r[1])
                self.output['sequences'][-1]['distance(km)'] += self.d[r[0], r[1]]
            dia += 1
        self.output['total_time_in_hours'] = sum(sequence['time(h)'] for sequence in self.output['sequences'])
        self.output['total_number_of_days'] = dia - 1
        self.output['average_working_time'] = self.output['total_time_in_hours'] / dia - 1
        file = open('backup_output.txt', 'w')
        file.write(str(self.output))
        print('Objeto de resposta do modelo:')
        print(self.output)

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

    def plot_solution(self):
        from itertools import cycle
        cycol = cycle('bgrcmk')
        x = [self.coords[i][0] for i in self.client_nodes]
        y = [self.coords[i][1] for i in self.client_nodes]
        fig, ax = plt.subplots()
        ax.plot(self.coords[0][0], self.coords[0][1], 'o')
        ax.plot(x, y, 'o')
        x = [self.coords[i][0] for i in self.all_nodes]
        y = [self.coords[i][1] for i in self.all_nodes]
        for arcs in self.arcs_sequence:
            color = next(cycol)
            for arc in arcs:
                self.connectpoints(x, y, arc[0], arc[1], color)
        plt.show()

    def save_on_google(self):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
            gc = gspread.authorize(credentials)
        except:
            print('Erro na conexcao')
            exit()
        wks = gc.open(self.spreadsheet_name).get_worksheet(1)
        i = 2
        columns = ['id', 'fantasia', 'NOMECLIENTE', 'ENDERECO', 'dia', 'ordem']
        for day in self.output['sequences']:
            j = 1
            for node in day['sequence']:
                if node != 0:
                    cell_list = wks.range(i, 1, i, 6)
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
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        try:
            credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
            gc = gspread.authorize(credentials)
        except:
            print('Erro na conexcao')
            exit()
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

    def connectpoints(self, x, y, p1, p2, color):
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
    'tempo_atendimento': 0.4,
    'tempo_km': 1 / 40,
    'horas_dia': 8
}

if __name__ == '__main__':
    instance = TSPModel(input_data)
    instance.start_point = (-3.7897703, -38.6155416)
    instance.spreadsheet_name = 'CARTEIRA DE CLIENTES AREA 18  JULHO 18'
    instance.get_data()
    instance.solve_partitioned(timelimit=600, partition='four')
    instance.generate_results()
    instance.plot_solution()
