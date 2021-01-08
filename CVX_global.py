import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations
from classifier import classifier as cf
import matplotlib.pyplot as plt
from failed.Optimal_solutions import read_female, read_male




agents = 20

relaxation = 100

tr = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
te = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

tr_m = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
te_m = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

for i in range(agents):#checheck check
    j = i * 16
    k = (i + 1) * 16

    train, test = read_female(j, k)
    tr = tr.append(train, ignore_index=True)
    te = te.append(test, ignore_index=True)

    train_m, test_m = read_male(j, k)
    tr_m = tr_m.append(train_m, ignore_index=True)
    te_m = te_m.append(test_m, ignore_index=True)


X_tr = tr.iloc[:, 1:]
Y_tr = tr.loc[:, 'CVD']

X_te = te.iloc[:, 1:]
Y_te = te.loc[:, 'CVD']

X_trm = tr_m.iloc[:, 1:]
Y_trm = tr_m.loc[:, 'CVD']

X_tem = te_m.iloc[:, 1:]
Y_tem = te_m.loc[:, 'CVD']


X_tr = X_tr.values
X_te = X_te.values

Y_tr = Y_tr.values
Y_te = Y_te.values

Y_tr = Y_tr[:, None]
Y_te = Y_te[:, None]


X_trm = X_trm.values
X_tem = X_tem.values

Y_trm = Y_trm.values
Y_tem = Y_tem.values

Y_trm = Y_trm[:, None]
Y_tem = Y_tem[:, None]

W = cp.Variable((4, 1))
b = cp.Variable()
xi = cp.Variable((len(X_tr), 1), nonneg=True)

loss = cp.sum(xi)
reg = cp.square(cp.norm(W))
obj_func = cp.Minimize(0.5 * reg + relaxation * loss)

Bi = [cp.multiply(Y_tr[i], X_tr[i] @ W + b) >= 1 - xi[i] for i in range(len(X_tr))]



prob = cp.Problem(obj_func, Bi)

prob.solve()

plt.style.use('seaborn')

w = W.value
b = b.value

cost_reduction, ROC_Curve, ROC_Score = cf(X_te, Y_te, w, b)
fpr, tpr = ROC_Curve
plt.plot(fpr, tpr, color='red')
print(ROC_Score)
print(cost_reduction)
print('\n')


Wm = cp.Variable((4, 1))
bm = cp.Variable()
xim = cp.Variable((len(X_trm), 1), nonneg=True)

loss = cp.sum(xim)
reg = cp.square(cp.norm(Wm))
obj_func = cp.Minimize(0.5 * reg + relaxation * loss)
Bim = [cp.multiply(Y_trm[i], X_trm[i] @ Wm + bm) >= 1 - xim[i] for i in range(len(X_trm))]
probm = cp.Problem(obj_func, Bim)

probm.solve()

wm = Wm.value
bm = bm.value

cost_reductionm, ROC_Curvem, ROC_Scorem = cf(X_tem, Y_tem, wm, bm)
fprm, tprm = ROC_Curvem
plt.plot(fprm, tprm, color='blue')

print(ROC_Scorem)
print(cost_reductionm)

plt.title('Area Under the ROC Curves - Optimal Solutions')
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Rate')
plt.show()






