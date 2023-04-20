# ---------------------------------------------------------------
# Anonymous ICME2023 Authors, Paper-id:541
# ---------------------------------------------------------------
exp=$1
gpu=$2
echo "export CUDA_VISIBLE_DEVICES=$gpu"
export CUDA_VISIBLE_DEVICES=${gpu}
nohup python run_experiments.py --exp $exp >  run_exp_$exp.out  
