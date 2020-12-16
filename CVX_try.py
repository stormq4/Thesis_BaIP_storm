import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male4.csv", delimiter=";", sep="\n"
                   ,dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})

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

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_train = np.array(y_train)
y_train.shape = (20, 1)
#y_test = y_test.values
#y_test = np.array(y_test)
#y_test.shape = (20, 1)

#print(X_train)
#print(X_train.shape[1])
#print(y_train)
print(X_train)


var = 4
W = cp.Variable((var, 1))
b = cp.Variable()

loss = cp.sum(cp.pos(1 - cp.multiply(y_train, X_train @ W + b)))

I = np.identity(var)
relaxation = 1
reg = cp.quad_form(W, I)
prob = cp.Problem(cp.Minimize(relaxation * loss + 0.5 * reg))

prob.solve(solver=cp.SCS, verbose=True, use_indirect=True)
print("optimal value with SCS:{}".format(prob.value))
print("W* is \n{} ".format(W.value))
print("b* is {}".format(b.value))

#print(X_test[4])
#print(prob.solve())
#print(W.value)
#print(b.value)
#print(prob.status)

y_test_predicted = np.array([])
weight = W.value
weight = np.array(weight)
weight.shape = (1, var)
weight = str(weight)[1:-1]
#weight = np.array(weight, dtype=float)

#print(weight)

#for i in range(X_test.shape[0]):
    #yp = np.sign(np.dot(W.value, X_test[i]))  # model
    #y_test_predicted = np.append(y_test_predicted, yp)




