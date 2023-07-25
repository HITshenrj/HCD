import numpy as np
import networkx as nx
from scipy.stats import norm
import os
from castle.datasets import DAG, IIDSimulation
for curDir, dirs, files in os.walk("nonlinear_data"):
    if dirs==[] :
        number=int(curDir.split('\\')[1])
        print(number)
        weighted_random_dag = DAG.erdos_renyi(n_nodes=number, n_edges=number*number*0.2, weight_range=(0.5, 2.0), seed=int(curDir.split('\\')[2]))
        dataset = IIDSimulation(W=weighted_random_dag, n=2000, method='nonlinear', sem_type='mlp')
        np.save(curDir+'/data.npy',dataset.X)
        np.save(curDir+'/truth.npy',dataset.B)