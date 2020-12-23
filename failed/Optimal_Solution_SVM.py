from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from failed.SVM_cost import sgd
from failed.Global_Cost_Function import global_cost
import time

def init(file):
    #reading csv and replacing 0 with -1
    data = pd.read_csv(file)
    data['CVD'].replace({0.0: -1.0}, inplace=True)

    #defining target value Y and X values for training and testing
    Y = data.loc[:, 'CVD']  # all rows of 'diagnosis'
    X = data.iloc[:, 2:]  # all rows of column 1 and ahead (features)


    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # first insert 1 in every row for intercept b
    X.insert(loc=len(X.columns), column='intercept', value=1)

    #dividing train and test samples
    X_train = X[0:4000]
    X_test = X[4000:8000]
    y_train = Y[0:4000]
    y_test = Y[4000:8000]

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

def results_df( Acc, AUC, CR):
    name = {
        'Accuracy': Acc,
        'AUC-ROC': AUC,
        'Cost Reduction': CR
    }
    return name

#dataframe for results loal lex-optimal ssolutions
def X_i(Xi, cost, gc):
    name = {
        'X_i*': Xi,
        'Cost': cost,
        'Global Cost': gc
    }
    return name


res_data = []

#X_i* and cost
X_ir = []


#test
if __name__ == '__main__':
    start_time = time.time()

    X_im, cost_m, Accm, AUCM, CRM = init(r"/Data/Tr_Te/male.csv")
    X_if, cost_f, Accf, AUCF, CRF = init(r"/Data/Tr_Te/female.csv")

    gcm = global_cost(X_im)
    gcf = global_cost(X_if)

    res_data.append(results_df(Accm, AUCM, CRM))
    res_data.append(results_df(Accf, AUCF, CRF))

    X_ir.append(X_i(X_im, cost_m, gcm))
    X_ir.append(X_i(X_if, cost_f, gcf))

    results = pd.DataFrame(res_data)
    results.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/Optimal_Results.csv')

    X_i_results = pd.DataFrame(X_ir)
    X_i_results.to_csv(r'/Users/stormdequay/PycharmProjects/pythonProject/results/X_i_Optimal.csv')

    end_time = time.time()

    time_r = end_time - start_time

    print('time is {}'.format(time_r))

