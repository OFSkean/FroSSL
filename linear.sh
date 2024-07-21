if [ $# -eq 0 ]
then
    echo "No arguments supplied. exiting"
    exit 1
fi

EXPERIMENT_NAME="$1"
DATASET="imagenet100"
CONFIG_NAME="frossl.yaml"
WANDBID="hnbezzo1"
TRAINED_CHECKPOINT_PATH="/home/AD/ofsk222/Research/clones/solo-learn/trained_models/frossl/hnbezzo1/frossl-imagenet100-hnbezzo1-ep=399.ckpt"

# check if data is cifar
if [ "$DATASET" == "cifar10" ] || [ "$DATASET" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$DATASET"
fi
LINEAR_CONFIG_PATH="scripts/linear/$dataset_config_name"

echo "Preparing to start linear probe for experiment with name $EXPERIMENT_NAME"
echo "on dataset $DATASET"
echo "with config $LINEAR_CONFIG_PATH/$CONFIG_NAME"

touch last_ckpt.txt
echo $TRAINED_CHECKPOINT_PATH > last_ckpt.txt

# ####
# #### PRETRAIN LINEAR PROBE
# ####
CUDA_LAUNCH_BLOCKING=1 python3 -u main_linear.py \
    --config-path $LINEAR_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++name="$EXPERIMENT_NAME-linear-$WANDBID" \
    ++data.dataset="$DATASET" \