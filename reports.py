import gspread
from oauth2client.service_account import ServiceAccountCredentials
import ast
from os import listdir
from os.path import isfile, join
from pprint import pprint



scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
try:
    credentials = ServiceAccountCredentials.from_json_keyfile_name("Legislator-b96aaa67134d.json", scope)
    gc = gspread.authorize(credentials)
except:
    print('Erro na conexcao')
    exit()


def build_dict(seq, key):
    return dict((d[key], dict(d, index=index)) for (index, d) in enumerate(seq))


def report_1(gc):
    mypath = './results'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if f[-1] == 't']
    onlyfiles.sort()
    sps = gc.open('Rotas - EstarBem')
    for i in range(len(onlyfiles)):

        file = open('./results/{}'.format(onlyfiles[i]), 'r')
        print(file)
        output = ast.literal_eval(file.read())
        sales_man_id = onlyfiles[i].split('_')[1]
        try:
            wks = sps.add_worksheet(title=sales_man_id, rows=200, cols=100)
        except:
            wks = sps.worksheet(sales_man_id)
        cell_list = wks.range('A1:D1')
        labels = ['ENTIDADEID', 'DESCRICAO', 'ROTAID', 'SEQUÊNCIA']
        for label, cell in zip(labels, cell_list):
            cell.value = label
        print('upddating head------------')
        wks.update_cells(cell_list)
        clients = build_dict(output['clients'], 'id')
        cell_list = wks.range(2, 1, len(output['clients']) + 2, 4)
        sequence = []
        for s in output['sequences']:
            sequence += s['sequence'][1:]
        day = 1
        i = 1
        j = 0
        order = 1
        for cell in cell_list:
            try: 
                if sequence[j] == 0:
                    order = 1
                    j += 1
                    day += 1
                if i == 1:
                    cell.value = clients[sequence[j]]['entidade_id']
                    i += 1
                elif i == 2:
                    cell.value = clients[sequence[j]]['description']
                    i += 1
                elif i == 3:
                    cell.value = day
                    i += 1
                else:
                    cell.value = order
                    i = 1
                    j += 1
                    order += 1
            except:
                pass
        wks.update_cells(cell_list)

def report_2(gc):
    mypath = './results'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) if f[-1] == 't']
    onlyfiles.sort()
    sps = gc.open('resultados das rotas')
    for i in range(18, len(onlyfiles)):
        file = open('./results/{}'.format(onlyfiles[i]), 'r')
        output = ast.literal_eval(file.read())
        sales_man_id = onlyfiles[i].split('_')[1]
        wks = sps.worksheet('Coordendas_incoerentes')


report_1(gc)


