#!/bin/sh
source ~/.bashrc
conda activate solo-learn
cd ../..

CONFIG=$1
FILENAME=$2
BATCH_SIZE=$3
PROJ_DIM=$4
NUM_AUGS=$5
HYDRA_FULL_ERROR=1 
echo "Running $CONFIG with batch size $BATCH_SIZE and projection dimension $PROJ_DIM"
echo "and $NUM_AUGS augmentations"

python3 -u main_pretrain.py \
    --config-path scripts/pretrain/stl10 \
    --config-name $CONFIG \
    ++name="$FILENAME" \
    ++wandb.enabled="False" \
    ++profiler.enabled="True" \
    ++profiler.strategy="advanced" \
    ++profiler.dirpath="experiments/time_scaling/logs" \
    ++profiler.filename="$FILENAME" \
    ++augmentations.0.num_crops="$NUM_AUGS" \
    ++max_epochs=1 \
    ++checkpoint.enabled="False" \
    ++method_kwargs.proj_output_dim="$PROJ_DIM" \
    ++optimizer.batch_size="$BATCH_SIZE"