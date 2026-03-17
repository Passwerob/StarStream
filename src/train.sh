#使用 4，5，6，7四张卡进行运行
cd /share/magic_group/aigc/fcr/EventVGGT/data/StreamVGGT_mixed_FSDP/src

CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 \
accelerate launch --use_fsdp --num_processes 4 --main_process_port 29667 \
./train.py --config-name train_dl3dv_fsdp