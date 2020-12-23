import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male4.csv", delimiter=";", sep="\n"
                   ,dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})

df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
data1 = df[0:8000]

#print(data1)
#print(data1)

data1['CVD'].replace({0.0: -1.0}, inplace=True)


Y = data1.loc[:, 'CVD']  # all rows of 'CVD'
X = data1.iloc[:, 1:]  # all rows of column 1 and ahead (features)


X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

#print(X)
#print(Y)

X_train = X[0:4000]
X_test = X[4000:8000]
y_train = Y[0:4000]
y_test = Y[4000:8000]

#print(y_train)

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_train = np.array(y_train)
y_train.shape = (4000, 1)
#y_test = y_test.values
#y_test = np.array(y_test)
#y_test.shape = (20, 1)

#print(X_train)
#print(X_train.shape[1])
#print(y_train)
#print(X_train)

var = 4
W = cp.Variable((var, 1))
b = cp.Variable()

#eps1 = cp.Variable((X_train.shape[0]))

#eps = cp.Variable()
#eps >= cp.pos(1 - cp.multiply(y_train, X_train @ W + b))

eps = cp.pos(1 - cp.multiply(y_train, X_train @ W + b))
loss = cp.sum(eps)



I = np.identity(var)
relaxation = 1
reg = cp.quad_form(W, I)
opt = cp.Minimize(0.5 * reg + relaxation * loss)
constraints = [opt == 17.190251792913706]#[eps1 >= 0]
prob = cp.Problem(opt)#, constraints)#, constraints)

#prob.solve()
prob.solve(solver=cp.SCS, verbose=True, use_indirect=True)
print("optimal value with SCS:{}".format(prob.value))
print("W* is \n{} ".format(W.value))
print("b* is {}".format(b.value))