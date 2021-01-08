import matplotlib.pyplot as plt
from classifier import classifier as cf
from CVX_male03 import agent_list
from dataframe import local_results
import pandas as pd

agents = 20


#hier worden de agents geinitialiseerd

colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'brown', 'pink', 'lime', 'navy', 'cyan',
           'salmon', 'purple', 'grey', 'olive', 'chocolate', 'cadetblue', 'cornsilk']

plt.style.use('seaborn')
#local solutions for every node

local_res = []

for i in range(agents):
    agent = agent_list[i]

    w = agent.W.value
    b = agent.b.value

    cost_reduction, ROC_Curve, ROC_Score = cf(agent.x_test, agent.y_test, agent.W.value, agent.b.value)
    fpr, tpr = ROC_Curve

    plt.plot(fpr, tpr, color=colours[i])
    local_res.append(local_results(i, ROC_Score, cost_reduction))

    #store solutions in files
    #f(x), x in een file en auc_roc score in de andere
plt.title('Area Under the ROC Curves - Male')
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Rate')
plt.show()

lm = pd.DataFrame(local_res)
lm.to_csv(r'results2/local_male.csv')