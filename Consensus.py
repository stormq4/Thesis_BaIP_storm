import pandas as pd
import numpy as np
from Global_Cost_Function import global_cost
import matplotlib.pyplot as plt

consensus = False

nodes = 20

class network_01:
    def __init__(self, node, neighbours, c_male, c_female, check):  # , X_i_male, X_i_female): #, X_i_star, direct, cons):
        self.node = node
        self.neighbours = neighbours
        self.c_male = c_male
        self.c_female = c_female
        self.check = check
        # self.X_i_male = X_i_male
        # self.X_i_female = X_i_female
        # self.X_i_star = X_i_star        self.direct = direct        self.cons = cons


class network_03:
    def __init__(self, node, neighbours, c_male, c_female, check):#, X_i_male, X_i_female): #, X_i_star, direct, cons):
        self.node = node
        self.neighbours = neighbours
        self.c_male = c_male
        self.c_female = c_female
        self.check = check
        #self.X_i_male = X_i_male
        #self.X_i_female = X_i_female
        #self.X_i_star = X_i_star        self.direct = direct        self.cons = cons

X_i_male = pd.read_csv(r'results/X_Node_male.csv', delimiter=",", sep="\n", usecols=['X_i*'])
X_i_female = pd.read_csv(r'results/X_Node_female.csv', delimiter=",", sep="\n", usecols=['X_i*'])

cost_male = pd.read_csv(r'results/X_Node_male.csv', delimiter=",", sep="\n", usecols=['Cost'])
cost_female = pd.read_csv(r'results/X_Node_female.csv', delimiter=",", sep="\n", usecols=['Cost'])

#print(cost_male.iloc[0, 0])

#X_i_male = np.array(X_i_male, dtype= np.float64)
#X_i_female = np.array(X_i_female)

edges_01 = pd.read_csv(r'Network/network_01_new.csv', delimiter=" ", sep="\n", header=None)
edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)

data_frame = []
for i in range(nodes):
    globals()['n_01_%s' % i] = network_01(i, [], cost_male.iloc[i, 0], cost_female.iloc[i, 0], [])#, X_i_male[i], X_i_female.iloc[i])
    globals()['n_03_%s' % i] = network_03(i, [], cost_male.iloc[i, 0], cost_female.iloc[i, 0], []) #, X_i_male[i], X_i_female.iloc[i])



    fill_df = []
    fill_df.append('N_%s' % i)
    data_frame.extend(fill_df)

    globals()['dr_n_%s' % i] = []



#print(n_01_4.c_male)
print(data_frame[19])
for i in range(edges_01.shape[0]):
    nb = []
    nb.append(edges_01.iloc[i, 1])
    for j in range(nodes):
        if globals()['n_01_%s' % j].node == edges_01.iloc[i, 0]:
            globals()['n_01_%s' % j].neighbours.extend(nb)

for i in range(edges_03.shape[0]):
    nb = []
    nb.append(edges_03.iloc[i, 1])
    for j in range(nodes):
        if globals()['n_03_%s' % j].node == edges_03.iloc[i, 0]:
            globals()['n_03_%s' % j].neighbours.extend(nb)


loop = 0
check03 = 0

distance = 0
optimal = 5710.9489204432775



while check03 < 20:
    for i in range(nodes):
        #if globals()['n_03_%s' % i].node == i:
        distance_check = []
        d = 0
        distance = optimal - globals()['n_03_%s' % i].c_male
        #distance_check.append(distance)
        distance_check.append(distance)

        globals()['dr_n_%s' % i].extend(distance_check)

        #print('distance in node {} is {}'.format(i, distance))
        for j in range(len(globals()['n_03_%s' % i].neighbours)):

            if globals()['n_03_%s' % globals()['n_03_%s' % i].neighbours[j]].c_male < globals()['n_03_%s' % i].c_male:
                #compute value here can be anything
                globals()['n_03_%s' % i].c_male = globals()['n_03_%s' % globals()['n_03_%s' % i].neighbours[j]].c_male
                distance = optimal - globals()['n_03_%s' % i].c_male


            elif globals()['n_03_%s' % globals()['n_03_%s' % i].neighbours[j]].c_male == globals()['n_03_%s' % i].c_male:
                d += 1
                distance = optimal - globals()['n_03_%s' % i].c_male

            if d == len(globals()['n_03_%s' % i].neighbours):
                globals()['n_03_%s' % i].check = 1
    check1 = 0
    for j in range(nodes):
        if globals()['n_03_%s' % j].check == 1:
            check03 += 1
    loop += 1

data = {'round': range(len(globals()['dr_n_%s' % 1]))}
d_male_03 = pd.DataFrame(data)

colours = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'orange', 'brown', 'pink', 'lime', 'navy', 'cyan',
           'salmon', 'purple', 'grey', 'olive', 'chocolate', 'cadetblue', 'cornsilk']

for i in range(nodes):
    #print(globals()['dr_n_%s' % i])
    d_male_03['{}'.format(data_frame[i])] = globals()['dr_n_%s' % i]
print(d_male_03)

plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

num = 0
for column in d_male_03.drop('round', axis=1):

    plt.plot(d_male_03['round'], d_male_03[column], marker='', color = colours[num], linewidth=1, alpha=0.9, label=column)
    num += 1

print(num)

plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Male G(n=20, p=0.3)", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Communication Round")
plt.ylabel("Error Distance")
plt.legend(loc='lower right', fontsize = 11, ncol=2)
plt.show()

#print(globals()['n_03_%s' % globals()['n_03_%s' % 5].neighbours[2]].c_male)
#print(loop)

#print(globals()['n_01_%s' % 10].neighbours)

check01 = 0
loop1 = 0
distance = 0
while check01 < 20:
    for i in range(nodes):
        #if globals()['n_03_%s' % i].node == i:
        d = 0
        distance = optimal - globals()['n_01_%s' % i].c_male
        #print('distance in node {} is {} for p=0.1'.format(i, distance))
        for j in range(len(globals()['n_01_%s' % i].neighbours)):

            if globals()['n_01_%s' % globals()['n_01_%s' % i].neighbours[j]].c_male < globals()['n_01_%s' % i].c_male:
                #compute value here can be anything
                globals()['n_01_%s' % i].c_male = globals()['n_01_%s' % globals()['n_01_%s' % i].neighbours[j]].c_male
                distance = optimal - globals()['n_01_%s' % i].c_male

            elif globals()['n_01_%s' % globals()['n_01_%s' % i].neighbours[j]].c_male == globals()['n_01_%s' % i].c_male:
                d += 1
                distance = optimal - globals()['n_01_%s' % i].c_male

            if d == len(globals()['n_01_%s' % i].neighbours):
                globals()['n_01_%s' % i].check = 1

    #check = 0

    for j in range(nodes):
        if globals()['n_01_%s' % j].check == 1:
            check01 += 1
    loop1 += 1

#print(loop1)


for i in range(nodes):
    break
    #print(globals()['n_03_%s' % i].c_male)
    #print(i)
#print(edges_03.iloc[1,1])

#print(n_03_15.neighbours[4])

#r = global_cost(n_03_15.X_i_male)
#print(r)





