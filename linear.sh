if [ $# -eq 0 ]
then
    echo "No arguments supplied. exiting"
    exit 1
fi

EXPERIMENT_NAME="$1"
ENTROPY_CUTOFF="$2"
TRAINED_CHECKPOINT_PATH="$3"
DATASET="stl10"
CONFIG_NAME="mmcr.yaml"

LINEAR_CONFIG_PATH="scripts/linear/$DATASET"
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '-' '{print $4}')

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
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID" \