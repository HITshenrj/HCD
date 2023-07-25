In our experiment,we use different parameters.You can use python run.py to run our algorithm and --***=*** to modify the parameters.Our default parameters are just for the Linear Gaussion data in 'data'.If you want to have the same result with us ,please read our experiment introduction.
If you want to learn the detail about sub-graph algorithms,please read the introduction about gcastle.

data:
Our Linear Gaussion data simulator is in utils.py and Our Nonlinear data is generated in utils_nolinaer.py by gcastle.
Our experiment data is in 'data'(Linear Gaussion) and 'nonlinear_data'(Nonlinear Mlp).You can use python --data_path=***/**/* to select the data to run.

example:
Linear Gaussion:
python run.py --method='ICALiNGAM' --data_path=data/100/1
python run.py --method='DirectLiNGAM' --data_path=data/100/1
python run.py --method='PC' --data_path=data/100/1
python run.py --method='GraNDAG' --data_path=data/100/1
python baseline_run.py --method='ICALiNGAM' --data_path=data/100/1
python baseline_run.py --method='DirectLiNGAM' --data_path=data/100/1
python baseline_run.py --method='PC' --data_path=data/100/1
python baseline_run.py --method='GraNDAG' --data_path=data/100/1


Nonlinear Mlp:
python run.py --method='ICALiNGAM' --data_path=nonlinear_data/100/1 --thresh=0.1 --pre_gate=0.97
python run.py --method='DirectLiNGAM' --data_path=nonlinear_data/100/1 --thresh=0.1 --pre_gate=0.97
python run.py --method='Notears' --data_path=nonlinear_data/100/1 --thresh=0.3 --pre_gate=0.97
python run.py --method='PC' --data_path=nonlinear_data/100/1 --pre_gate=0.97
python baseline_run.py --method='ICALiNGAM' --data_path=nonlinear_data/100/1 --thresh=0.1
python baseline_run.py --method='DirectLiNGAM' --data_path=nonlinear_data/100/1 --thresh=0.1
python baseline_run.py --method='Notears' --data_path=nonlinear_data/100/1 --thresh=0.1
python baseline_run.py --method='PC' --data_path=nonlinear_data/100/1