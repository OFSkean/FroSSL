#!/bin/sh

cd ../..

CONFIG=$1
FILENAME=$2
BATCH_SIZE=$3
PROJ_DIM=$4
NUM_AUGS=$5

echo "$CONFIG - BS $BATCH_SIZE - DIM $PROJ_DIM - AUGS $NUM_AUGS"

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
    ++max_epochs=5 \
    ++checkpoint.enabled="False" \
    ++method_kwargs.proj_hidden_dim="$PROJ_DIM" \
    ++method_kwargs.proj_output_dim="$PROJ_DIM" \
    ++optimizer.batch_size="$BATCH_SIZE" \
    ++precision="16-mixed" \
    ++data.num_workers=16