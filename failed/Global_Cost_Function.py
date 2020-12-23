import numpy as np
import pandas as pd

def global_cost(W, X, Y, reg_strength): #REMOVE XYand reg_strength
    # calculate global cost
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = reg_strength * (np.sum(distances) / N)

    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost


if __name__ == "__main__":
    #results_male = pd.read_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_Node_male_good.csv', delimiter=",", sep="\n", dtype= {'X_i*' : np.float64})

    results_male = pd.read_csv(r'../results/X_Node_male.csv', usecols=['X_i*'], delimiter=",", sep="\n")
    xi_star = pd.read_csv(results_male, delimiter=' ', sep='\n', dtype=np.float64)
    df = pd.DataFrame(xi_star)

    nodes = 20
    #for i in nodes:
        #compute_cost(results_male.iloc[i, 2])

    #print(compute_cost(results_male.iloc[1, 2]))

    #rm = results_male.iloc[1, 2]

    print(df)

    #cost = 1 / 2 * np.dot(rm, rm)

    #print(cost)

    #print(results_male['X_i*'])

    #print(compute_cost(results_male[2, 'X_i*']))