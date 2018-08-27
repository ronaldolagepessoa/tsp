from coopr.pyomo import *
from pyomo.opt import SolverStatus, TerminationCondition
import calendar
from datetime import date, timedelta


def modelo_enfermeiras(entrada):

    d1 = date(entrada['anoDeInicio'], entrada['mesDeInicio'], entrada['diaDeInicio'])
    d2 = date(entrada['anoDeTermino'], entrada['mesDeTermino'], entrada['diaDeTermino'])
    delta = d2 - d1

    week_days = []
    weekend_days = []
    all_days = [d for d in range(d1.day, d2.day + 1)]
    for i in range(delta.days + 1):
        data = d1 + timedelta(days=i)
        if calendar.weekday(data.year, data.month, data.day) <= 4:
            week_days.append(data.day)
        else:
            weekend_days.append(data.day)

    E = [enfermeira['id'] for enfermeira in entrada['enfermeiras']]
    K = [0, 1, 2]
    l = entrada['regimeDeFolga']

    model = ConcreteModel()

    model.u = {(e['id']): [] for e in entrada['enfermeiras']}
    for e in entrada['enfermeiras']:
        if e['diaDeInicioFerias'] != None:
            for d in all_days:
                if d < e['diaDeInicioFerias'] or d > e['diaDeTerminoFerias']:
                    model.u[e['id']].append(d)
        else:
            model.u[e['id']] += all_days

    model.pt = {(e['id'], k): 1 if e['preferenciaDeTurno'] == k else 0 for e in entrada['enfermeiras'] for k in K}
    model.b = {(demanda['dia'], demanda['turno']): demanda['valor'] for demanda in entrada['demandas']}
    model.h = {enfermeira['id']: enfermeira['horas'] for enfermeira in entrada['enfermeiras']}

    model.z = Var(E, within=Binary)
    model.y = Var(E, week_days, [0, 1], within=Binary)
    model.x = Var(E, weekend_days, within=Binary)
    model.w = Var(E, all_days, within=Binary)
    model.alpha = Var(E, K, within=Binary)

    model.obj = Objective(
        expr=sum((1 - model.pt[e, k]) * model.alpha[e, k] for e in E for k in K),
        sense=minimize
    )
    model.c1 = ConstraintList()
    for e in E:
        model.c1.add(
            6 * sum(model.y[e, d, k] for d in week_days for k in [0, 1]) +
            12 * (
                    sum(model.x[e, d] for d in weekend_days) +
                    sum(model.w[e, d] for d in all_days)
            ) <= model.h[e]
        )
    model.c2 = ConstraintList()
    for d in week_days:
        for k in [0, 1]:
            model.c2.add(
                sum(model.y[e, d, k] for e in E if d in model.u[e]) >= model.b[d, k]
            )
    model.c3 = ConstraintList()
    for d in weekend_days:
        for k in [0, 1]:
            model.c3.add(
                sum(model.x[e, d] for e in E if d in model.u[e]) >= model.b[d, k]
            )
    model.c4 = ConstraintList()
    for d in all_days:
        model.c4.add(
            sum(model.w[e, d] for e in E if d in model.u[e]) >= model.b[d, 2]
        )
    model.c5 = ConstraintList()
    for e in E:
        for d in week_days:
            for k in [0, 1]:
                model.c5.add(
                    model.y[e, d, k] <= model.alpha[e, k]
                )
    model.c6 = ConstraintList()
    for e in E:
        for i in range(len(weekend_days) - 1):
            if weekend_days[i + 1] - weekend_days[i] == 1:
                model.c6.add(
                    model.x[e, weekend_days[i]] + model.x[e, weekend_days[i + 1]] <=
                    sum(model.alpha[e, k] for k in K)
                )
    model.c7 = ConstraintList()
    for e in E:
        for i in range(len(all_days) - l):
            model.c7.add(
                sum(model.w[e, all_days[i + j]] for j in range(l + 1)) <=
                model.alpha[e, 2]
            )
    model.c8 = ConstraintList()
    for e in E:
        for d in weekend_days:
            model.c8.add(
                model.w[e, d] + model.x[e, d] <=
                model.z[e]
            )
    model.c9 = ConstraintList()
    for e in E:
        model.c9.add(
            sum(model.alpha[e, k] for k in K) <= model.z[e]
        )


    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        output = {'error': False, 'escalas': [{'dia': d, 'enfermeiras': []} for d in all_days]}
        for e in E:
            for d in week_days:
                if model.y[e, d, 0].value > 0:
                    output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 0, 'valor': 'M'})
                if model.y[e, d, 1].value > 0:
                    output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 1, 'valor': 'T'})
            for d in weekend_days:
                if model.x[e, d].value > 0:
                    if model.alpha[e, 0].value > 0:
                        output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 0, 'valor': 'MT'})
                    elif model.alpha[e, 1].value > 0:
                        output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 1, 'valor': 'MT'})
                    else:
                        output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 2, 'valor': 'MT'})
            for d in all_days:
                if model.w[e, d].value > 0:
                    output['escalas'][d - 1]['enfermeiras'].append({'id': e, 'turno': 2, 'valor': 'SN'})

        return output
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        return {'error': True, 'msg': 'Não existe solução viável para o modelo.'}
    else:
        return {'error': True, 'msg': 'Problema na solução do modelo.'}


def modelo_tecnicos(entrada):
    d1 = date(entrada['anoDeInicio'], entrada['mesDeInicio'], entrada['diaDeInicio'])
    d2 = date(entrada['anoDeTermino'], entrada['mesDeTermino'], entrada['diaDeTermino'])
    delta = d2 - d1

    week_days = []
    weekend_days = []
    all_days = [d for d in range(d1.day, d2.day + 1)]
    for i in range(delta.days + 1):
        data = d1 + timedelta(days=i)
        if calendar.weekday(data.year, data.month, data.day) <= 4:
            week_days.append(data.day)
        else:
            weekend_days.append(data.day)

    T = [tecnico['id'] for tecnico in entrada['tecnicos']]
    K = [0, 1, 2]
    l = entrada['regimeDeFolga']

    model = ConcreteModel()

    model.u = {(t['id']): [] for t in entrada['tecnicos']}
    for t in entrada['tecnicos']:
        if t['diaDeInicioFerias'] != None:
            for d in all_days:
                if d < t['diaDeInicioFerias'] or d > t['diaDeTerminoFerias']:
                    model.u[t['id']].append(d)
        else:
            model.u[t['id']] += all_days

    model.pt = {(t['id'], k): 1 if t['preferenciaDeTurno'] == k else 0 for t in entrada['tecnicos'] for k in K}

    model.b = {(demanda['dia'], demanda['turno']): demanda['valor'] for demanda in entrada['demandas']}

    model.x = Var(T, weekend_days, within=Binary)
    model.w = Var(T, all_days, within=Binary)
    model.alpha = Var(T, K, within=Binary)

    model.obj = Objective(
        expr=sum((1 - model.pt[t, k]) * model.alpha[t, k] for t in T for k in K),
        sense=minimize
    )
    model.c2 = ConstraintList()
    for d in week_days:
        for k in [0, 1]:
            model.c2.add(
                sum(model.alpha[t, k] for t in T if d in model.u[t]) >= model.b[d, k]
            )
    model.c3 = ConstraintList()
    for d in weekend_days:
        for k in [0, 1]:
            model.c3.add(
                sum(model.x[t, d] for t in T if d in model.u[t]) >= model.b[d, k]
            )
    model.c4 = ConstraintList()
    for d in all_days:
        model.c4.add(
            sum(model.w[t, d] for t in T if d in model.u[t]) >= model.b[d, 2]
        )
    model.c6 = ConstraintList()
    for t in T:
        for i in range(len(weekend_days) - 1):
            if weekend_days[i + 1] - weekend_days[i] == 1:
                model.c6.add(
                    model.x[t, weekend_days[i]] + model.x[t, weekend_days[i + 1]] <=
                    sum(model.alpha[t, k] for k in [0, 1])
                )
    model.c7 = ConstraintList()
    for t in T:
        for i in range(len(all_days) - l):
            model.c7.add(
                sum(model.w[t, all_days[i + j]] for j in range(l + 1)) <=
                model.alpha[t, 2]
            )
    model.c8 = ConstraintList()
    for t in T:
        model.c8.add(
            sum(model.alpha[t, k] for k in K) <= 1
        )

    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=True)

    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        output = {'error': False, 'escalas': [{'dia': d, 'tecnicos': []} for d in all_days]}
        for t in T:
            for d in week_days:
                if d in model.u[t]:
                    if model.alpha[t, 0].value > 0:
                        output['escalas'][d - 1]['tecnicos'].append({'id': t, 'turno': 0, 'valor': 'M'})
                    if model.alpha[t, 1].value > 0:
                        output['escalas'][d - 1]['tecnicos'].append({'id': t, 'turno': 1, 'valor': 'T'})
            for d in weekend_days:
                if d in model.u[t]:
                    if model.x[t, d].value > 0:
                        if model.alpha[t, 0].value > 0:
                            output['escalas'][d - 1]['tecnicos'].append({'id': t, 'turno': 0, 'valor': 'MT'})
                        else:
                            output['escalas'][d - 1]['tecnicos'].append({'id': t, 'turno': 1, 'valor': 'MT'})
            for d in all_days:
                if d in model.u[t]:
                    if model.w[t, d].value > 0:
                        output['escalas'][d - 1]['tecnicos'].append({'id': t, 'turno': 2, 'valor': 'SN'})

        return output
    elif (results.solver.termination_condition == TerminationCondition.infeasible):
        return {'error': True, 'msg': 'Não existe solução viável para o modelo.'}
    else:
        return {'error': True, 'msg': 'Problema na solução do modelo.'}






