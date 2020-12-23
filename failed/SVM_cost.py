import numpy as np
from sklearn.utils import shuffle

#computing local cost
def compute_cost(W, X, Y, reg_strength):
    # calculate relaxation
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

#calculating cost gradient to compute cost
def calculate_cost_gradient(W, X_batch, Y_batch, reg_strength):
    #filtering values for cost gradient
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

#computes cost
def sgd(features, outputs, reg_strength, learning_rate):
    #epochs to solve the local SVM
    max_epochs = 2000 #change this value for testing

    weights = np.zeros(features.shape[1])
    nth = 0
    prev_cost = float("inf")
    cost_threshold = 0.00005# in percent

    # stochastic gradient descent
    for epoch in range(1, max_epochs):

        #X, Y = features, outputs
        X, Y = shuffle(features, outputs)

        for ind, x in enumerate(X):
            ascent = calculate_cost_gradient(weights, x, Y[ind], reg_strength)

            weights = weights - (learning_rate * ascent)
        # convergence check on 2^nth epoch

        if epoch == 2 ** nth or epoch == max_epochs - 1:
            cost = compute_cost(weights, features, outputs, reg_strength)

            #testing
            print("Epoch is:{} and Cost is: {}".format(epoch, cost))
            # stoppage criterion

            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights, cost

            prev_cost = cost
            nth += 1

    return weights, cost