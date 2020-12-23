from disropt.agents import Agent
from disropt.communicators import MPICommunicator
from disropt.algorithms import Consensus
import pandas as pd
import numpy as np

edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)
nodes = 20


comm = MPICommunicator()

for i in range(nodes):
    in_neighb = []
    out_neighb = []

    for j in range(edges_03.shape[0]):
        if edges_03.iloc[j, 1] == i:
            in_neighb.append(edges_03.iloc[j, 0])

    for j in range(edges_03.shape[0]):

        if edges_03.iloc[j, 0] == i:
            out_neighb.append(edges_03.iloc[j, 1])



    print(out_neighb)
    vect = np.random.rand(2, 1)

    globals()['agent-%s' % i] = Agent(in_neighb, out_neighb)

    globals()['exch_data_%s' %i] = comm.neighbors_exchange(vect, in_neighb, out_neighb, dict_neigh=False)





