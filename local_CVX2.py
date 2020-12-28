import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#deze file solved voor elke 'node' of 'agent' in het netwerk met de cvxpy solver
#Ik denk dat mijn cost functie sowieso niet klopt met variables

#elke node krijgt een class
class agent_03:
    def __init__(self, node, in_nb, out_nb, W, b, eps):#, X_i_male, X_i_female): #, X_i_star, direct, cons):
        self.node = node
        self.in_nb = in_nb
        self.out_nb = out_nb
        self.W = W
        self.b = b
        self.eps = eps

agents = 20

#dit is om data te lezen van een csv file
edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)

data = pd.read_csv(r"Data/male4.csv", delimiter=";", sep="\n"
                   ,dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})
df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

#Maakt mijn data een binary variable
df['CVD'].replace({0.0: -1.0}, inplace=True)

Y = df.loc[:, 'CVD']  # all rows of 'CVD'
X = df.iloc[:, 1:]

X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

relaxation = 100
points_per_agent = 200

var = 4

#defining van variables
W = cp.Variable((var, 1))
b = cp.Variable()
eps = cp.Variable((4000, 1), nonneg=True)

loss = cp.sum(eps)

reg = cp.square(cp.norm(W))

#cost function, kloopt waarschijnlijk niet
opt = cp.Minimize(0.5 * reg + relaxation * loss)

#loop om mijn agents te vullen met neighbours en
X_train_total = np.zeros((1,4))
X_test_total = np.zeros((1,4))
y_test_total = np.zeros((1,1))
y_train_total = np.zeros((1,1))


for i in range(agents):



    X_train = X[400 * i: 400 * i + 200]
    X_test = X[400 * i + 200: 400 * (i + 1)]
    y_train = Y[400 * i: 400 * i + 200]
    y_test = Y[400 * i + 200: 400 * (i + 1)]

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_train = np.array(y_train)
    y_train.shape = (y_train.shape[0], 1)
    y_test = np.array(y_test)
    y_test.shape = (y_test.shape[0], 1)

    X_train_total = np.concatenate((X_train_total, X_train), axis=0)
    X_test_total = np.concatenate((X_test_total, X_test), axis=0)
    y_test_total = np.concatenate((y_test_total, y_test), axis=0)
    y_train_total = np.concatenate((y_train_total, y_train), axis=0)



X_train_total = X_train_total[1:,:]
X_test_total = X_test_total[1:,:]
y_test_total = y_test_total[1:,:]
y_train_total = y_train_total[1:,:]

constraints = [cp.multiply(y_train_total, X_train_total @ W + b) >= 1 - eps]
prob = cp.Problem(opt, constraints)

prob.solve()

print("optimal value with SCS for node {} is :{}".format(0, prob.value))
print("W* is \n{} ".format(W.value))
print("b* is {}".format(b.value))

print()
#eigenlijk al mijn waardes die uit mijjn cost functieon rollen zijn ruk dus klopt waarschijnlijk helemaal niets van haha
#succes ;)