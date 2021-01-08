import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from classifier import classifier as cf
import matplotlib.pyplot as plt
from CVX_male03 import agent_list

iterations = 1

# Dit had de time iteration moeten worden, maar omdat de basis niet berekend kon worden, is dat dus niet af
for t in range(iterations):
    #store previously computed values of Bi

    store_base = []
    for agent in agent_list:
        agent.Bt = agent.Bi
        store_base.append(agent.Bt)

    #computing new base
    for agent in agent_list:
        # initialising bt and retrieving from neighbours
        # with constraints from Bi, Xi and Bj
        # and compute new Bi
        Bt = []
        Xi = agent.Xi

        Bj = []
        B = None
        for nb in agent.in_nb:
            B = store_base[nb]
            Bj.append(B)

        Bt.append(Xi)
        Bt.append(agent.Bt)
        Bt.append(Bj)
        Bt.append(Bj)
        #agent.Bi = Bt

        #if t > iterations - 5:
            #print(agent.node)
            #print(agent.W.value)
            #print(agent.b.value)
            #print(t)