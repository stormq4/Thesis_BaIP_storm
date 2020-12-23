import networkx as nx
from network_SVM import create_network
from failed.SVM_female import init_f
import pandas as pd
import numpy as np
from failed.SVM_male import init_m
from failed.Global_Cost_Function import global_cost
import time


#reading male data and put in file
def reader_writer1(file, begin, end, node_file): #, begin, end): #reads file and
    data = pd.read_csv(file, delimiter=";", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,
                                                             'di_bp': np.float64, 'chol': np.float64})
    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df[begin:end].to_csv(node_file)
    return df[begin:end]

#reading female data and put in file
def reader_writer2(file, begin, end, node_file): #, begin, end): #reads file and
    data = pd.read_csv(file, delimiter=",", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,
                                                             'di_bp': np.float64, 'chol': np.float64})
    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df[begin:end].to_csv(node_file)
    return df[begin:end]

#dataframe for results auc score, etc
def results_df(i, Acc, AUC, CR):
    name = {
        'Accuracy': Acc,
        'Node': i,
        'AUC-ROC': AUC,
        'Cost Reduction': CR
    }
    return name

#dataframe for results loal lex-optimal ssolutions
def X_i(i, Xi, cost, gc):
    name = {
        'Node': i,
        'X_i*': Xi,
        'Cost': cost,
        'Global Cost': gc
    }
    return name

#creating network with n = 20 nodes and p probability
n = 20
p = 0.1
g = create_network(n, p)

#node, acc, cost reduction
res_data_m = []
res_data_f = []

#X_i* and cost
X_i_male = []
X_i_female = []

#assigning data to nodes and retrieving local lex-optimal solutions
for i in g.nodes:
    start_time = time.time()
    j = i*400
    k = (i+1)*400
    male = reader_writer1(r"../Data/male4.csv", j, k,
                         r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_male/mnode_%s.csv" % i)
    female = reader_writer2(r"/Data/female4.csv", j, k,
                           r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_female/fnode_%s.csv" % i)

    attrs = {i: {"male": male, "female": female}}
    nx.set_node_attributes(g, attrs)

    X_im, cost_m, Accm, AUCM, CRM = init_m(r'/Users/stormdequay/PycharmProjects/pythonProject/Data/node_male/mnode_%s.csv' %i, i)
    X_if, cost_f, Accf, AUCF, CRF = init_f(r'/Users/stormdequay/PycharmProjects/pythonProject/Data/node_female/fnode_%s.csv' %i, i)

    res_data_m.append(results_df(i, Accm, AUCM, CRM))
    res_data_f.append(results_df(i, Accf, AUCF, CRF))

    gcm = global_cost(X_im)
    gcf = global_cost(X_if)

    X_i_male.append(X_i(i, X_im, cost_m, gcm))
    X_i_female.append(X_i(i, X_if, cost_f, gcf))

    #print(gcf)
    #print('node %', i)

    end_time = time.time()

    time_node = end_time - start_time

    print('time in node {} is {}'.format(i, time_node))


#putting results from local SVm in files
results_m = pd.DataFrame(res_data_m)
results_m.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/results_male_Acc_CR.csv')

results_f = pd.DataFrame(res_data_f)
results_f.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/results_female_Acc_CR.csv')


X_node_male = pd.DataFrame(X_i_male)
X_node_male.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_male.csv')

X_node_female = pd.DataFrame(X_i_female)
X_node_female.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_female.csv')

