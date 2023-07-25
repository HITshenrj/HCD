import numpy as np
from castle.metrics import MetricsDAG
from model import SAHCD
import time
from args import get_args
"""
args:
--method:Algorithm for sub-graph.You can choose:ICALiNGAM(default),DirectLiNGAM,PC,Notears(GOLEM),GraNDAG
--pre_gate:Threshold for conditional independence(fisherz test) tests(default=0.8).Higher thresholds represent higher confidence requirements
--thresh:Threshold for sub-graph(default=0.3).
--golem_epoch:Iteration rounds of the GOLEM(default=5000).
--lr:Learning rate for GOLEM(default=0.05).
--pc_alpha:Parameter of PC(default =0.05).
--data_path:Parameter of data path(default=data/100/0).
The parameter of our algorithm is pre_gate.You can get detailed descriptions of other parameters from gcastle.
"""
if __name__ =='__main__':
    args=get_args()
    print(args.data_path)
    print(args)
    data = np.load(args.data_path+'/data.npy')
    Tdata = np.load(args.data_path+'/truth.npy')
    Tdata= np.int64(Tdata != 0)
    start=time.time()
    model=SAHCD(data,Tdata,args)
    model.run()
    end=time.time()
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write(str(args) + '\t'+str(end-start)+'\t')
        f.write(str(MetricsDAG(model.global_graph, Tdata).metrics) + '\n')