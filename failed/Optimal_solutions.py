import pandas as pd
import numpy as np


def read_male(j, k):
    data = pd.read_csv(r"Data/male4.csv", delimiter=";", sep="\n", dtype={'CVD': np.float64, 'BMI': np.float64,
                                                                   'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})

    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df['CVD'].replace({0.0: -1.0}, inplace=True)
    return df[j: k - 8], df[k - 8: k]

def read_female(j, k):
    data = pd.read_csv(r"Data/female4.csv", delimiter=",", sep="\n", dtype={'CVD': np.float64, 'BMI': np.float64,
                                                                          'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})

    df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
    df['CVD'].replace({0.0: -1.0}, inplace=True)
    return df[j: k - 8], df[k - 8: k]



tr_m = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
te_m = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

tr_f = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
te_f = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

male = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])
female = pd.DataFrame(columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

for i in range(20):#checheck check
    j = i * 16
    k = (i + 1) * 16

    train_male, test_male = read_male(j, k)
    tr_m = tr_m.append(train_male, ignore_index=True)
    te_m = te_m.append(test_male, ignore_index=True)

    train_female, test_female = read_female(j, k)
    tr_f = tr_f.append(train_female, ignore_index=True)
    te_f = te_f.append(test_female , ignore_index=True)



train_m = pd.DataFrame(tr_m)
train_m.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/tr_m.csv")


test_m = pd.DataFrame(te_m)
test_m.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/te_m.csv")


male = train_m.append(test_m, ignore_index=True)
male.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/male.csv")

train_f = pd.DataFrame(tr_f)
train_f.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/tr_f.csv")

test_f = pd.DataFrame(te_f)
test_f.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/te_f.csv")

female = train_f.append(test_f, ignore_index=True)
female.to_csv(r"/Users/stormdequay/PycharmProjects/pythonProject/Data/Tr_Te/female.csv")

