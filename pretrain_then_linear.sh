DATASET="${1:-stl10}"  #first argument is dataset, default is stl10
CONFIG_NAME="${2:-searmse.yaml}" #second argument is dataset, default is searmse
PRETRAIN_CONFIG_PATH="scripts/pretrain/$DATASET"
LINEAR_CONFIG_PATH="scripts/linear/$DATASET"

echo $CONFIG_NAME
echo $PRETRAIN_CONFIG_PATH
echo $LINEAR_CONFIG_PATH

####
#### PRETRAIN BACKBONE
####
python3 -u main_pretrain.py \
    --config-path $PRETRAIN_CONFIG_PATH \
    --config-name $CONFIG_NAME \

# get pretrained path from last_ckpt.txt file
TRAINED_CHECKPOINT_PATH=$(cat last_ckpt.txt)
echo "$TRAINED_CHECKPOINT_PATH"

####
#### PRETRAIN LINEAR PROBE
####
python3 -u main_linear.py \
    --config-path $LINEAR_CONFIG_PATH \
    --config-name $CONFIG_NAME \