EXPERIMENT_NAME="frossl-4view-weighted-0.6"
ENTROPY_CUTOFF="0.6"
TRAINED_CHECKPOINT_PATH="/home/AD/ofsk222/Research/clones/solo-learn/trained_models/4view_decay_sweep/decay-0.6/frossl-4view-weighted-0.6-tjy85uxa-ep=499.ckpt"
DATASET="stl10"
CONFIG_NAME="searmse.yaml"

LINEAR_CONFIG_PATH="scripts/linear/$DATASET"
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '-' '{print $7}')

echo "Preparing to start linear probe for experiment with name $EXPERIMENT_NAME"
echo "on dataset $DATASET"
echo "with config $CONFIG_NAME"
echo "with entropy cutoff $ENTROPY_CUTOFF"
echo "checkpoint path $TRAINED_CHECKPOINT_PATH"
echo "checkpoint wandbid $TRAINED_CHECKPOINT_WANDB_ID"

echo $TRAINED_CHECKPOINT_PATH > last_ckpt.txt

# ####
# #### PRETRAIN LINEAR PROBE
# ####
CUDA_LAUNCH_BLOCKING=1 python3 -u main_linear.py \
    --config-path $LINEAR_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID"
#    ++method_kwargs.entropy_cutoff=$ENTROPY_CUTOFF


