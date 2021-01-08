import matplotlib.pyplot as plt
from classifier import classifier as cf
from CVX_female03 import agent_list
import pandas as pd
from dataframe import local_results

agents = 20
colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'brown', 'pink', 'lime', 'navy', 'cyan',
           'salmon', 'purple', 'grey', 'olive', 'chocolate', 'cadetblue', 'cornsilk']

plt.style.use('seaborn')

local_res = []
#local solutions for every node
for i in range(agents):
    agent = agent_list[i]

    w = agent.W.value
    b = agent.b.value

    cost_reduction, ROC_Curve, ROC_Score = cf(agent.x_test, agent.y_test, w, b)
    fpr, tpr = ROC_Curve

    plt.plot(fpr, tpr, color=colours[i])
    local_res.append(local_results(i, ROC_Score, cost_reduction))

plt.title('Area Under the ROC Curves - Female')
plt.ylabel('True Positive Rate')
plt.xlabel('False Negative Rate')
plt.show()

lf = pd.DataFrame(local_res)
lf.to_csv(r'results2/local_female.csv')