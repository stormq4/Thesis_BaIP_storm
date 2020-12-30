import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from itertools import combinations

#deze file solved voor elke 'node' of 'agent' in het netwerk met de cvxpy solver
#Ik denk dat mijn cost functie sowieso niet klopt met variables

#elke node krijgt een class
class Agent:
    def __init__(self, node, x_train, x_test, y_train, y_test):
        self.node = node
        self.in_nb = []
        self.out_nb = []
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        # Bt is Basis op dit moment
        self.Bt = None

        # d is het aantal variabelen, gelijk aan lengte van de tweede as van X
        self.d = len(x_train[0, :])
        self.W = cp.Variable((self.d, 1))
        self.b = cp.Variable()
        self.xi = cp.Variable((len(x_train), 1), nonneg=True)
        print()
        # list comprehension van alle constraints
        # Bi is initial Basis
        self.Bi = [cp.multiply(self.y_train[i], self.x_train[i] @ self.W + self.b) >= 1 - self.xi[i] for i in range(len(x_train))]

        loss = cp.sum(self.xi)
        reg = cp.square(cp.norm(self.W))
        self.obj_func = cp.Minimize(0.5 * reg + relaxation * loss)
        base_problem = cp.Problem(self.obj_func, self.Bi)
        base_problem.solve()
        # Huidige optimum waarde
        self.value = [self.W.value, self.b.value]
        self.constraints = self.Bi

        # Berekening van de basis op basis van constraints
        self.compute_basis()


    def compute_basis(self):
        for possible_basis in combinations(self.constraints, self.d + 1):
            # convert candidate basis to list
            possible_basis = list(possible_basis)

            # los het op met mogelijke basis
            basis_prob = cp.Problem(self.obj_func, possible_basis)
            basis_prob.solve()
            basis = self.constraints
            try:
                if self.W.value == self.value[0] and self.b.value == self.value[1]:
                    # basis gevonden
                    basis = possible_basis
                    break
            except:
                pass

        self.constraints = basis

agents = 20
n_agents = 20

#dit is om data te lezen van een csv file
edges_03 = pd.read_csv(r'Network/network_03_new.csv', delimiter=" ", sep="\n", header=None)

data = pd.read_csv(r"Data/male4.csv", delimiter=";", sep="\n"
                   ,dtype={'CVD': np.float64,'BMI': np.float64,'sys_bp': np.float64,'di_bp': np.float64, 'chol': np.float64})
df = pd.DataFrame(data, columns=['CVD', 'BMI', 'sys_bp', 'di_bp', 'chol'])

#Maakt mijn data een binary variable
df['CVD'].replace({0.0: -1.0}, inplace=True)

Y = df.loc[:, 'CVD']  # all rows of 'CVD'
X = df.iloc[:, 1:]

X_normalized = MinMaxScaler().fit_transform(X.values)
X = pd.DataFrame(X_normalized)

relaxation = 100
points_per_agent = 200

var = 4

iterations = 1000

#X en Y worden naar numpy arrays omgezet
X = X.values
Y = Y.values
#Het aantal datapunten is gelijk aan de lengte van de eerste as van X (de tweede as zijn namelijk bmi, chol, etc)
n_datapoints = len(X[:,0])

print()

#hier worden de agents geinitialiseerd
agent_list = []
for i in range(n_agents):
    # het aantal punten per agent is gelijk aan totaal datapunten delen door aantal agents, en delen door twee
    # vanwege train / test
    pp_agents = int(n_datapoints / n_agents / 2)

    # X en Y worden gesliced zodat je per agent de goede data hebt
    agent_x_train = X[2*i*pp_agents:(2*i+1)*pp_agents,:]
    agent_x_test = X[(2*i+1)*pp_agents:(2*i+2)*pp_agents,:]
    # Laatste deel voegt een lege axis toe, zodat dimension van (4,) naar (4,1) gaat
    # Hierdoor kan cvxpy ze samen met X gebruiken
    agent_y_train = Y[2 * i * pp_agents:(2 * i + 1) * pp_agents][:, None]
    agent_y_test = Y[(2 * i + 1) * pp_agents:(2 * i + 2) * pp_agents][:, None]
    agent = Agent(i, agent_x_train, agent_x_test, agent_y_train, agent_y_test)
    for j in range(edges_03.shape[0]):
        if edges_03.iloc[j, 1] == i:
            agent.in_nb.append(edges_03.iloc[j, 0])

    for j in range(edges_03.shape[0]):
        if edges_03.iloc[j, 0] == i:
            agent.out_nb.append(edges_03.iloc[j, 1])
    agent_list.append(agent)
print()
# Dit had de time iteration moeten worden, maar omdat de basis niet berekend kon worden, is dat dus niet af
for t in range(iterations):
    for agent in agent_list:
        Bi = agent.Bi


