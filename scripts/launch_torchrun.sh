# WARNING: This script is incompatible with the utility-replay (Method 1) and
# online-Fisher plasticity decay (Method 2) mechanisms. It launches multiple
# DDP ranks but SleepSampler is not rank-sharded and the Fisher accumulator is
# not cross-rank all-reduced. Running it will silently produce wrong numbers.
# Use 'python train.py ...' directly until multi-GPU support is added.
cd ..
host_ip=$(hostname --ip-address)
port_number=$(shuf -i 29510-49510 -n 1)
torchrun --nnodes 1 --rdzv_endpoint $host_ip:$port_number --nproc_per_node 2 train.py $@
