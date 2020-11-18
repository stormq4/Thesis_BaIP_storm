from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import numpy as np

def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

# I haven't tested it but this same function should work for
# vanilla and mini-batch gradient descent as well
def calculate_cost_gradient(W, X_batch, Y_batch):
    # if only one example is passed (eg. in case of SGD)
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    dw = np.zeros(len(W))
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            di = W
        else:
            di = W - (reg_strength * Y_batch[ind] * X_batch[ind])
        dw += di
    dw = dw/len(Y_batch)  # average

    return dw

def sgd(features, outputs):
    max_epochs = 20
    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.0005  # in percent
    # stochastic gradient descent
    for epoch in range(1, max_epochs):
        # shuffle to prevent repeating update cycles
        X, Y = features, outputs
        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind])
            weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch

        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs)
            #print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights
            prev_cost = cost
            nth += 1

    return weights

def init_m(file, j):
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
    #print("splitting dataset into train and test sets...")



    # train the model
    #print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    #print("training finished.")
    #print("weights are: {}".format(W))

    # print(y_test_predicted)
    #print(X_train)
    # print(X_test)
    # testing the model on test set

    y_test_predicted = np.array([])
    count_CVD_correct = 0
    count_NO_correct = 0
    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.to_numpy()[i]))  # model
        y_test_predicted = np.append(y_test_predicted, yp)

        #if y_test_predicted == 1.0 and y_test[i, 'CVD'] == 1.0:
         #   count_CVD_correct += 1

    #for i in range(len(y_test_predicted)):
       # if y_test_predicted == 1.0 and y_test[i] == 1.0:
          #  count_CVD_correct += 1

    for v in enumerate(y_test):
        if v[1] == 1.0 and y_test_predicted[v[0]] == 1.0:
            count_CVD_correct += 1

    #print(count_CVD_correct)

    #print(y_test[:, 'CVD'])
    #print(y_test[2, 'CVD'])
    node_cost_reduction = cost_reduction * count_CVD_correct
    #print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), y_test_predicted)))
    #print("recall on test dataset: {}".format(recall_score(y_test.to_numpy(), y_test_predicted)))
    #print("precision on test dataset: {}".format(precision_score(y_test.to_numpy(), y_test_predicted)))
    #print("cost reduction of node %s: " %j, "{}" .format(node_cost_reduction), " euros per QALY\n")

    return W, accuracy_score(y_test.to_numpy(), y_test_predicted), node_cost_reduction

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
    #print("splitting dataset into train and test sets...")



    # train the model
    #print("training started...")
    W = sgd(X_train.to_numpy(), y_train.to_numpy())
    #print("training finished.")
    #print("weights are: {}".format(W))

    # print(y_test_predicted)
    #print(X_train)
    # print(X_test)
    # testing the model on test set

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


reg_strength = 1000 # regularization strength
learning_rate = 0.00000001
cost_reduction = 5100

if __name__ == '__main__':
    init_m(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/male_44/male441.csv", 1)
