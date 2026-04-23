#!/bin/bash
#SBATCH --nodes=2               # number of nodes
#SBATCH --partition=primary     # GPU partition
#SBATCH --ntasks-per-node=4     # number of tasks per node
#SBATCH --cpus-per-task=8       # number of cpus requested per task
#SBATCH --threads-per-core=1    # number of threads per core
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --time=12:00:00         # format: HH:MM:SS
##SBATCH --nodelist=gpu002,gpu004
##SBATCH --mem 0
#SBATCH --job-name=ddpslurm
#SBATCH -o workspace/%x.output%j
#SBATCH -e workspace/%x.error%j
##SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=none
#SBATCH --overcommit
#SBATCH --parsable

#--nodes=2  --ntasks-per-node=4 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:4 
#--nodes=2  --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:8 -missing gpus from node on timeline view
#--nodes=8  --ntasks-per-node=1 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:1 -works
#--nodes=4  --ntasks-per-node=2 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:2 -works
#--nodes=2  --ntasks-per-node=2 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:2 -works
#--nodes=1  --ntasks-per-node=8 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:8 
#--nodes=4  --ntasks-per-node=4 --cpus-per-task=8 --threads-per-core=1 --gres=gpu:4 

echo "NODELIST="${SLURM_NODELIST}
echo "WORLD_SIZE="${SLURM_NPROCS}

# QdstrmImporter -i multinode_nvtx.qdstrm 
# load conda env

#module load conda/2023
#conda init
#source activate env_mnode

module load nsight-systems/2024.5.1

#export TMPDIR=$PWD/workspace/reports/tmp
#export TMPDIR=workspace/reports/tmp
#echo $TMPDIR
echo $PWD

# get master node IP Address 

SERVER="`hostname`"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST"))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
export MASTER_PORT=12340
echo "Master IP address: " $MASTER_ADDR

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}

# Force Ring algorithm if needed
#export NCCL_ALGO=Ring
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=INIT,GRAPH,TOPO,TUNING,COLL,TRACE
#NCCL_TOPO_DUMP_FILE=reports/topo_ring.xml

#export CUDA_VISIBLE_DEVICES=0,1,2,3
cd workspace
# --output reports/multinode_nvtx_resnet50_opt%q{SLURM_NODEID}_%q{SLURM_PROCID} \
srun nsys profile --trace cuda,osrt,nvtx,ucx \
--gpu-metrics-devices=cuda-visible \
--cuda-flush-interval=0 \
--nic-metrics=true \
--output reports/multinode_nvtx_resnet50_opt%q{SLURM_NODEID}_%q{SLURM_PROCID} \
--force-overwrite true /home/$USER/.venv/bin/python source_code/slurm_ddp.py
#/home/$USER/.conda/envs/env_profiler/bin/python source_code/slurm_ddp.py
