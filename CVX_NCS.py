import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male4.csv", delimiter=";", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})

df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
data1 = df[0:40]

#print(data1)
#print(data1)

data1['CVD'].replace({0.0: -1.0}, inplace=True)


Y = data1.loc[:, 'CVD']  # all rows of 'CVD'
X = data1.iloc[:, 1:]  # all rows of column 1 and ahead (features)


X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

#print(X)
#print(Y)

X_train = X[0:20]
X_test = X[20:40]
y_train = Y[0:20]
y_test = Y[20:40]

#print(y_train)

#X_train = X_train.values
#X_train = np.array(X_train)
#X_train.shape = (20, 4)
#X_test = X_test.values
y_train = y_train.values
y_train = np.array(y_train)
y_train.shape = (20, 1)
#y_test = y_test.values
#y_test = np.array(y_test)
#y_test.shape = (20, 1)

#print(X_train[2])
#t = []
#t.append(X_train[2])
#print(t)
#print(X_train.shape[1])
#print(y_train)
#print(X_train)

nodes = X_train.shape[0]
#print(nodes)





edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)



class network_03:
    def __init__(self, node, neighbours, X, check):#, X_i_male, X_i_female): #, X_i_star, direct, cons):
        self.node = node
        self.neighbours = neighbours
        self.X = X
        self.check = check

#print(X_train[1])

for i in range(nodes):

    X = pd.DataFrame(X_train)
    X = X_train.loc[i]
    #X = X.to_numpy()
    X = X.values
    #print(X)
    #X = np.array(X)
    #X.shape = (1, 4)



    #print(X)
    #X = str(X)[1:-1]
    print(X)

    #X = float(X)

    globals()['n_03_%s' % i] = network_03(i, [], X, bool)

    #print(globals()['n_03_%s' % i].X)

for i in range(edges_03.shape[0]):
    nb = []
    nb.append(edges_03.iloc[i, 1])
    for j in range(nodes):
        if globals()['n_03_%s' % j].node == edges_03.iloc[i, 0]:
            globals()['n_03_%s' % j].neighbours.extend(nb)

var = 4


I = np.identity(var)
relaxation = 1


check03 = False

#while check03 == False:
for i in range(nodes):
    nb = []

    #nb1 = np.array(globals()['n_03_%s' % i].X)
    #nb = nb.append(nb1)

    nb.append(globals()['n_03_%s' % i].X)


    print(nb)


    #print(globals()['n_03_%s' % j].X)
    for j in range(len(globals()['n_03_%s' % i].neighbours)):
        #nb.append(globals()['n_03_%s' % j].X)
        #print(globals()['n_03_%s' % j].X)
        #print(nb)
        #nb.append(X_train[j])
        pass


    #nb.values
    #print(nb)
    #W = cp.Variable((var, 1))
    #b = cp.Variable()
    #loss = cp.sum(cp.pos(1 - cp.multiply(y_train, nb @ W + b)))
    #reg = cp.quad_form(W, I)
    #globals()['prob_N' % i] = cp.Problem(cp.Minimize(relaxation * loss + 0.5 * reg))
    #globals()['prob_N' % i].solve()
    #print("optimal value with SCS:{}".format(globals()['prob_N' % i].value))
    #print("W* is \n{} ".format(W.value))
    #print("b* is {}".format(b.value))


#print(n_03_14.neighbours)

#print(n_03_14.X)



