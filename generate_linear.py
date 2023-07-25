import numpy as np
import networkx as nx
from scipy.stats import norm
import os

def simulate_random_dag(d: int,
                        degree: float,
                        graph_type: str,
                        w_range: tuple = (0.5, 2.0)) -> nx.DiGraph:
    """Simulate random DAG.
    Args:
        d: variable number
        degree: DAG degree
        graph_type: {'erdos-renyi'}
        w_range: weight range

    Returns:
        G: random DAG
    """
    if graph_type == 'erdos-renyi':
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    else:
        raise ValueError('unknown graph type')
    # random permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def simulate_sem(G: nx.DiGraph,
                 n: int,
                 sem_type: str,
                 linear_type: str) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Args:
        G: weigthed DAG
        n: number of samples
        sem_type: {linear-gauss}
        linear_type:{linear,linear_sin,linear_tanh,linear_cos,linear_sigmoid}

    Returns:
        X: [n,d] sample matrix
    """
    def sigmoid(x):
        return 1/(1+np.exp(-x))
    W = nx.to_numpy_array(G)
    d = W.shape[0]
    if sem_type == 'linear-gauss':
        X = np.random.randn(n, d)
    else:
        raise ValueError('unknown sem type')
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    # X=aX+bf(X)+delta
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W[parents, j])
        if linear_type == 'linear':
            X[:, j] += eta
        elif linear_type == 'linear_sin':
            X[:, j] += 2.*np.sin(eta) + eta
        elif linear_type == 'linear_tanh':
            X[:, j] += 2.*np.tanh(eta) + eta
        elif linear_type == 'linear_cos':
            X[:, j] += 2.*np.cos(eta) + eta
        elif linear_type == 'linear_sigmoid':
            X[:, j] += 2.*sigmoid(eta) + eta
    return W,X
for curDir, dirs, files in os.walk("data"):
    if dirs==[] and files==[]:
        number=int(curDir.split('\\')[1])
        print(number)
        ground_truth=simulate_random_dag(number,int(number*0.4),'erdos-renyi')
        print(ground_truth)
        ground_truth,data=simulate_sem(ground_truth,1000,'linear-gauss','linear')
        print(ground_truth)
        print(data)
        np.save(curDir+'/data.npy',data)
        np.save(curDir+'/truth.npy',ground_truth)