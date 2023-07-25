import numpy as np
from castle.metrics import MetricsDAG
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
    method=args.method
    start = time.time()
    if method == 'ICALiNGAM':
        from castle.algorithms import ICALiNGAM
        model = ICALiNGAM(thresh=args.thresh)
    if method == 'DirectLiNGAM':
        from castle.algorithms import DirectLiNGAM
        model = DirectLiNGAM(thresh=args.thresh)
    if method == 'PC':
        from castle.algorithms import PC
        model = PC(alpha=args.pc_alpha)
    if method == 'Notears':
        from castle.algorithms import Notears
        model = Notears(w_threshold=args.thresh)
    if method == 'GraNDAG':
        from castle.algorithms import GraNDAG
        model = GraNDAG(input_dim=data.shape[1], device_type='gpu')
    if method == 'GOLEM':
        from castle.algorithms import GOLEM
        model = GOLEM(GOLEM(num_iter=args.golem_epoch, graph_thres=args.thresh, device_type='gpu',
                                learning_rate=args.lr))
    model.learn(data)
    end=time.time()
    print(MetricsDAG(model.causal_matrix, Tdata).metrics)
    with open('result_base.txt', 'a', encoding='utf-8') as f:
        f.write(str(args) + '\t'+str(end-start)+'\t')
        f.write(str(MetricsDAG(model.causal_matrix, Tdata).metrics) + '\n')