from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from SVM_cost import sgd

def init_m(file, j):
    #reading csv and replacing 0 with -1
    data = pd.read_csv(file, delimiter=";", sep="\n", dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,
                                                             'di_bp': np.float64, 'chol': np.float64})
    data['CVD'].replace({0.0: -1.0}, inplace=True)

    #defining target value Y and X values for training and testing
    Y = data.loc[:, 'CVD']  # all rows of 'diagnosis'
    X = data.iloc[:, 2:]  # all rows of column 1 and ahead (features)


    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    #dividing train and test samples
    X_train = X[0:200]
    X_test = X[200:400]
    y_train = Y[0:200]
    y_test = Y[200:400]

    W, cost = sgd(X_train.to_numpy(), y_train.to_numpy(), reg_strength, learning_rate)
    y_test_predicted = np.array([])
    count_CVD_correct = 0


    #counting values for classification
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)

    #checking correctly classifying men
    for v in enumerate(y_test):
        if v[1] == 1.0 and y_test_predicted[v[0]] == 1.0:
            count_CVD_correct += 1

    node_cost_reduction = cost_reduction * count_CVD_correct

    return W, cost, accuracy_score(y_test.to_numpy(), y_test_predicted), roc_auc_score(y_test.to_numpy(), y_test_predicted), node_cost_reduction


reg_strength = 10000 # regularization strength 170
learning_rate = 0.000001 #0.0000001
cost_reduction = 5100

#test
if __name__ == '__main__':
    w, c, a,auc , n = init_m(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_male/mnode_16.csv", 1)
    print(auc)