#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022"

# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

# python3 eval_on_3rscan.py --dataset_subset 3500 --euler True --eval_iter 20000000 
python3 train_on_3rscan.py \
    --euler True \
    --epochs 120 \
    --separate_cells_by_scene \
    --eval_iter 10000 \
    --model_name text2pos_NO_pretrained_10000_eval_iters