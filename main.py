import networkx as nx
from network_SVM import create_network
from SVM_female import init_f
import pandas as pd
import numpy as np
from SVM_male import init_m

#inseert cost

def reader_writer1(file, begin, end, node_file): #, begin, end): #reads file and
    data = pd.read_csv(file, delimiter=";", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,
                                                             'di_bp': np.float64, 'chol': np.float64})
    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df[begin:end].to_csv(node_file)
    return df[begin:end]

def reader_writer2(file, begin, end, node_file): #, begin, end): #reads file and
    data = pd.read_csv(file, delimiter=",", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,
                                                             'di_bp': np.float64, 'chol': np.float64})
    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df[begin:end].to_csv(node_file)
    return df[begin:end]

def results_df(i, Acc, CR):
    name = {
        'Node': i,
        'Accuracy': Acc,
        'Cost Reduction': CR
    }
    return name

def X_i(Xi, i, cost):
    name = {
        'Node': i,
        'X_i*': Xi,
        'Cost': cost
    }
    return name
n = 20
p = 0.1
g = create_network(n, p)

#node, acc, cost reduction
res_data_m = []
res_data_f = []

#X_i* and cost
X_i_male = []
X_i_female = []

for i in g.nodes:
    j = i*400
    k = (i+1)*400
    male = reader_writer1(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male4.csv", j, k,
                         r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_male/mnode_%s.csv" %i)
    female = reader_writer2(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/female4.csv", j, k,
                           r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_female/fnode_%s.csv" %i)

    attrs = {i: {"male": male, "female": female}}
    nx.set_node_attributes(g, attrs)

    X_im, cost_m, Accm, CRM = init_m(r'/Users/stormdequay/PycharmProjects/pythonProject/Data/node_male/mnode_%s.csv' %i, i)
    X_if, cost_f, Accf, CRF = init_f(r'/Users/stormdequay/PycharmProjects/pythonProject/Data/node_female/fnode_%s.csv' %i, i)

    res_data_m.append(results_df(i, Accm, CRM))
    res_data_f.append(results_df(i, Accf, CRF))

    X_i_male.append(X_i(X_im, i, cost_m))
    X_i_female.append(X_i(X_if, i, cost_f))

results_m = pd.DataFrame(res_data_m)
results_m.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/results_male_Acc_CR.csv')

results_f = pd.DataFrame(res_data_f)
results_f.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/results_female_Acc_CR.csv')


X_node_male = pd.DataFrame(X_i_male)
X_node_male.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_male.csv')
#X_node_male.to_excel(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_male.xlsx', index=None, header=True)

X_node_female = pd.DataFrame(X_i_female)
X_node_female.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_female.csv')