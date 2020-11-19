from sklearn.metrics import accuracy_score, recall_score
import pandas as pd
import numpy as np
from SVM_cost import sgd


def init_f(file, j):
    data = pd.read_csv(file)
    data['CVD'].replace({0.0: -1.0}, inplace=True)
    #data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
    # put features & outputs in different DataFrames for convenience
    Y = data.loc[:, 'CVD']  # all rows of 'diagnosis'
    X = data.iloc[:, 2:]  # all rows of column 1 and ahead (features)

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    X_train = X[0:200]
    X_test = X[200:400]
    y_train = Y[0:200]
    y_test = Y[200:400]

    W = sgd(X_train.to_numpy(), y_train.to_numpy(), reg_strength, learning_rate)


    y_test_predicted = np.array([])
    count_CVD_correct = 0
    count_NO_correct = 0
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)



    for v in enumerate(y_test):
        if v[1] == 1.0 and y_test_predicted[v[0]] == 1.0:
            count_CVD_correct += 1


    node_cost_reduction = cost_reduction * count_CVD_correct
    #print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    #print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    #print("precision on test dataset: {}".format(precision_score(y_test.to_numpy(), y_test_predicted)))
    #print("cost reduction of node %s: " %j, "{}" .format(node_cost_reduction), " euros per QALY\n")

    return W, accuracy_score(y_test.to_numpy(), y_test_predicted), node_cost_reduction


reg_strength = 100 # regularization strength 100
learning_rate = 0.00000001 #0.00000001
cost_reduction = 5100

if __name__ == '__main__':
    init_f(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/node_female/fnode_8.csv", 1)
