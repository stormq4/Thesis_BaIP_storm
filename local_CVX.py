import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class agent_03:
    def __init__(self, node, in_nb, out_nb, W, b, eps):#, X_i_male, X_i_female): #, X_i_star, direct, cons):
        self.node = node
        self.in_nb = in_nb
        self.out_nb = out_nb
        self.W = W
        self.b = b
        self.eps = eps

agents = 20

edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)

data = pd.read_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male4.csv", delimiter=";", sep="\n"
                   ,dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})
df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

df['CVD'].replace({0.0: -1.0}, inplace=True)

Y = df.loc[:, 'CVD']  # all rows of 'CVD'
X = df.iloc[:, 1:]

X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

relaxation = 10

var = 4

W = cp.Variable((var, 1))
b = cp.Variable()
eps = cp.Variable(nonneg=True)

loss = cp.sum(eps)

reg = cp.square(cp.norm(W))
opt = cp.Minimize(0.5 * reg + relaxation * loss)

for i in range(agents):
    in_neighb = []
    out_neighb = []

    for j in range(edges_03.shape[0]):
        if edges_03.iloc[j, 1] == i:
            in_neighb.append(edges_03.iloc[j, 0])

    for j in range(edges_03.shape[0]):
        if edges_03.iloc[j, 0] == i:
            out_neighb.append(edges_03.iloc[j, 1])

    X_train = X[400 * i: 400 * i + 200]
    X_test = X[400 * i + 200: 400 * (i + 1)]
    y_train = Y[400 * i: 400 * i + 200]
    y_test = Y[400 * i + 200: 400 * (i + 1)]

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_train = np.array(y_train)
    y_train.shape = (y_train.shape[0], 1)

    constraints = [cp.multiply(y_train, X_train @ W + b) >= 1 - eps]
    prob = cp.Problem(opt, constraints)

    prob.solve()

    globals()['n_03_%s' % i] = agent_03(i, in_neighb, out_neighb, W.value, b.value, eps.value)

    print("optimal value with SCS for node {} is :{}".format(i, prob.value))
    print("W* is \n{} ".format(W.value))
    print("b* is {}".format(b.value))
    print("eps is {} \n".format(eps.value))
    #print(prob.status)


