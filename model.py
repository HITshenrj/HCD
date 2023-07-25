import time
from scipy.stats import norm
from castle.metrics import MetricsDAG
from castle.algorithms import DirectLiNGAM
from castle.algorithms import Notears
from castle.algorithms import GOLEM
import numpy as np
from castle.algorithms import PC
from itertools import chain, combinations
from castle.algorithms import ICALiNGAM
"""
This class is used to judge casaul independence.
"""
class CondIndepParCorr():
    def __init__(self, data, n):
        super().__init__()
        self.correlation_matrix = np.corrcoef(data)
        self.num_records = n

    def calc_statistic(self, x, y, zz):
        corr_coef = self.correlation_matrix
        if len(zz) == 0:
            par_corr = corr_coef[x, y]
        elif len(zz) == 1:
            z = zz[0]
            par_corr = (
                (corr_coef[x, y] - corr_coef[x, z]*corr_coef[y, z]) /
                np.sqrt((1-np.power(corr_coef[x, z], 2))
                        * (1-np.power(corr_coef[y, z], 2)))
            )
        else:  # zz contains 2 or more variables
            all_var_idx = (x, y) + zz
            corr_coef_subset = corr_coef[np.ix_(all_var_idx, all_var_idx)]
            # consider using pinv instead of inv
            inv_corr_coef = -np.linalg.pinv(corr_coef_subset)
            par_corr = inv_corr_coef[0, 1] / \
                np.sqrt(abs(inv_corr_coef[0, 0]*inv_corr_coef[1, 1]))
        z = np.log1p(2*par_corr / (1-par_corr+1e-5))
        val_for_cdf = abs(
            np.sqrt(self.num_records - len(zz) - 3) *
            0.5 * z
        )
        statistic = 2*(1-norm.cdf(val_for_cdf))
        return statistic


class SAHCD:
    """
    self.data:Observation data
    self.Ture_data:ground_truth
    self.n:The number of samples
    self.dim:The number of Vertices
    self.cipc:FisherZ
    self.pre_set_gate:The threshold of fisherz
    self.global_graph:The global graph.
    self.colider_set:A set of colliders shared by subgraphs
    self.avg_time:Average time costed by each sub-graph
    self.max_time:Max time costed by each sub-graph
    self.args:Hyperparameters
    """
    def __init__(self,data,True_data,args):
        self.data=np.array(data)
        self.True_data=np.array(True_data)
        self.n=self.data.shape[0]
        self.dim=self.data.shape[1]
        self.cipc = CondIndepParCorr(self.data.T, self.n)
        self.IPset=[]
        self.pre_set_gate=args.pre_gate
        self.Sepset=[]
        self.global_graph=np.ones((self.dim,self.dim,),dtype=int)
        self.sepsets=[]
        self.colider_set={}
        self.avg_time=0
        self.max_time=0
        self.args=args
        for i in range (self.dim):
            self.global_graph[i][i]=0


    """
    run
    Get the sub-graphs and Use sub-graph algorithms to learn the causal graph.
    """
    def run(self):
        skeleton_truth = np.int64(self.True_data != 0)
        seperatesets=self.seperate_data()
        print(seperatesets)
        print(len(seperatesets))
        res = seperatesets[0]
        for i in seperatesets:
            if i == seperatesets[0]:
                continue
            res = res.intersection(i)
            with open('setsinf.csv', 'a') as f:
                f.write(str(res) + '\t')
        with open('setsinf.csv', 'a') as f:
            f.write('\n' + str(res) + '\n')
        self.sepsets = seperatesets
        self.colider_set = res
        for set in seperatesets:
            data=self.data[:,list(set)]
            self.get_Sep_IP(data,list(set))
        self.global_graph=np.int64(self.global_graph>0)
        final=MetricsDAG(self.global_graph,skeleton_truth)
        print(final.metrics)
        self.avg_time=self.avg_time/len(seperatesets)
    """
    seperate_data
    All nodes will be pruned using Fisherz's unconditional independence test. For each unshielded path, make them immorality.
    Args:
    No args.
    Returns:
    A family of sets.Each set in the set family is the node set of a sub-graph.
    """
    def seperate_data(self):
        for i in range (self.dim):
            for j in range (i+1,self.dim):
                i_j_conditions=tuple([])
                tmp = self.cipc.calc_statistic(i, j, i_j_conditions)
                # print(tmp)
                if tmp > self.pre_set_gate:
                    # print(tmp)
                    self.global_graph[i, j] = 0
                    self.global_graph[j, i] = 0
        for i in range (self.dim):
            for j in range (self.dim):
                for k in range (self.dim):
                    if self.global_graph[j,i]+self.global_graph[i,j]>0 and self.global_graph[k,i]+self.global_graph[i,k]>0 and self.global_graph[j,k]+self.global_graph[k,j]==0 and i!=j and j!=k and i!=k:
                        self.global_graph[i,j]=0
                        self.global_graph[i,k]=0
        seperate_sets=self.get_seperate_sets(self.global_graph)
        return seperate_sets

    """
    get_seperate_sets
    Args:
    A graph that performs unconditional independent pruning
    Returns:
    A family of sets containing subgraph cases. Each set in this set family is a set of all nodes of a subgraph
    """
    def get_seperate_sets(self,graph):
        seperate_sets = []
        hashset = np.zeros((self.dim))
        for i in range(self.dim):
            if hashset[i] == 0:
                hashset[i] = 1
                temp_set = set()
                _temp_set = set()
                _temp_set.add(i)
                while temp_set != _temp_set:
                    temp_set = _temp_set.copy()
                    for j in temp_set:
                        hashset[j] = 1
                        for k in range(self.dim):
                            if graph[j][k] == 1 and k not in temp_set:
                                _temp_set.add(k)
                seperate_sets.append(temp_set)

        final_sets = []

        for i in seperate_sets:
            flag = 0
            for j in seperate_sets:
                if i==j :
                    continue
                if i.issubset(j):
                    flag=1
                    break
            if flag==0:
                final_sets.append(i)


        return final_sets
    """
    Using subgraph algorithm and margin the algorithm by our rules.
    Args:
    data:The observation data of sub-graph.
    node:The list to matching node index between sub-graph and global graph.
    Returns:
    No returns but self.global_graph will be a causal graph with weight.
    """
    def get_Sep_IP(self,data,node):
        node=np.array(node)
        nums=data.shape[1]
        n = 1 << nums
        method=self.args.method
        # Choose sub-graph algorithm
        if method=='ICALiNGAM':
            sub_model = ICALiNGAM(thresh=self.args.thresh)
        if method=='DirectLiNGAM':
            sub_model = DirectLiNGAM(thresh=self.args.thresh)
        if method=='PC':
            sub_model = PC(alpha=self.args.pc_alpha)
        if method=='Notears':
            sub_model = Notears(w_threshold=self.args.thresh)
        if method=='GraNDAG':
            from castle.algorithms import GraNDAG
            sub_model = GraNDAG(input_dim=data.shape[1],device_type='gpu')
        if method=='GOLEM':
            sub_model=GOLEM(GOLEM(num_iter=self.args.golem_epoch,graph_thres=self.args.thresh,device_type='gpu',learning_rate=self.args.lr))
        start=time.time()
        sub_model.learn(data)
        cost_time=time.time()-start
        self.avg_time+=cost_time
        self.max_time=max(self.max_time,cost_time)
        for i in range(nums):
            for j in range(nums):
                if i not in self.colider_set or j not in self.colider_set:
                    if sub_model.causal_matrix[i][j] == 0:
                        self.global_graph[node[i]][node[j]] -= 10
                    else:
                        self.global_graph[node[i]][node[j]] += 10
                else:
                    if sub_model.causal_matrix[i][j] == 0:
                        self.global_graph[node[i]][node[j]]-=1
                    else:
                        self.global_graph[node[i]][node[j]]+=10000
        return