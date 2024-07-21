EXPERIMENT_NAME="wmse_eigs"
DATASET="stl10"  #second argument is dataset, default is stl10
CONFIG_NAME="wmse_eig" #third argument is dataset, default is searmse
NUM_AUGMENTATIONS="2"
BACKBONE="resnet18"



if [ "$DATASET" == "cifar10" ] || [ "$DATASET" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$DATASET"
fi

PRETRAIN_CONFIG_PATH="scripts/pretrain/$dataset_config_name"
LINEAR_CONFIG_PATH="scripts/pretrain/$dataset_config_name"


python3 -u main_pretrain.py \
    --config-path $PRETRAIN_CONFIG_PATH \
    --config-name $CONFIG_NAME \
    ++name="$EXPERIMENT_NAME" \
    ++backbone.name=$BACKBONE \
    ++data.dataset=$DATASET \