#!/bin/bash
# SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="/cluster/project/cvg/jiaqchen/Text2Pos-CVPR2022"

# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

python3 eval_on_3rscan.py \
    --euler True \
    --eval_iter 100 \
    --eval_iter_count 1000 \
    --separate_cells_by_scene \
    --eval_entire_dataset \
    --pretrained \
    --model_name text2pos_pretrained_no_validation_set_70_checkpoint