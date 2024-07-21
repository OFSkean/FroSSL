if [ $# -eq 0 ]
then
    echo "No arguments supplied. exiting"
    exit 1
fi

EXPERIMENT_NAME="$1"
DATASET="${2:-stl10}"  #second argument is dataset, default is stl10
CONFIG_NAME="${3:-searmse.yaml}" #third argument is dataset, default is searmse
NUM_AUGMENTATIONS="$4"
BACKBONE="resnet18"



if [ "$DATASET" == "cifar10" ] || [ "$DATASET" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$DATASET"
fi

PRETRAIN_CONFIG_PATH="scripts/pretrain/$dataset_config_name"
LINEAR_CONFIG_PATH="scripts/pretrain/$dataset_config_name"

echo "Preparing to start experiment with name $EXPERIMENT_NAME"
echo "on dataset $DATASET"
echo "with config $CONFIG_NAME"
echo "with entropy cutoff $ENTROPY_CUTOFF"
echo "with numaugs $NUM_AUGMENTATIONS"
echo "with backbone $BACKBONE"

####
#### PRETRAIN BACKBONE
####
python3 -u main_pretrain.py \
    --config-path $PRETRAIN_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++name="$EXPERIMENT_NAME" \
    ++backbone.name=$BACKBONE \
    ++data.dataset=$DATASET
    #++augmentations.0.num_crops=$NUM_AUGMENTATIONS \

# get pretrained path from last_ckpt.txt file
TRAINED_CHECKPOINT_PATH=$(cat last_ckpt.txt)
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '/' '{print $3}')
echo "$TRAINED_CHECKPOINT_WANDB_ID $TRAINED_CHECKPOINT_PATH"

####
#### PRETRAIN LINEAR PROBE
####
python3 -u main_linear.py \
    --config-path $LINEAR_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++data.dataset=$DATASET \
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID" \
