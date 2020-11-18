import networkx as nx
import matplotlib.pyplot as plt


def create_network(n,p):
    g = nx.erdos_renyi_graph(n, p, seed=None, directed=True)
    return g

if __name__ == '__main__':
    n = 20  # nodes of 20 hospitals
    p = 0.3 # probalilty of conectivity
    g = create_network(n,p)
    nx.draw(g, with_labels=True, node_color="r")
    plt.show()

    matrix = nx.adjacency_matrix(g)
    print(g.nodes)
    print(g.edges)
    #print(matrix.todense())
    if p == 0.1:
        nx.write_edgelist(g, '/Network/network_01_good.csv')
    if p == 0.3:
        nx.write_edgelist(g, '/Network/network_03_good.csv')


