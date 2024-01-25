#tradeoffs=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)

wandbids=("vicreg/jegp7rzs" "byol/230nmp0t" "mmcr/iple3ere" "wmse/eqhafr20" "swav/7s2n7pm6")
#wandbids=("dino/urt3o574" "corinfomax/axxtovsi" "searmse/wy0vxvcg" "barlow/4049ryr3")
model_folder_name="tiny_imagenet_method_sweep"
dataset="tiny-imagenet"
backbone="resnet50"
mkdir trained_models/${model_folder_name}

arraylength=${#wandbids[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    method_name=$(echo ${wandbids[i]} |cut -d/ -f 1)
    wandbid=$(echo ${wandbids[i]} |cut -d/ -f 2)
    experiment_name="$method_name-tiny"

    folder_prefix="$experiment_name"
    lcc_baseline_folder="/home/ofsk222/projects/frossl/solo-learn/trained_models/${wandbids[i]}"
    new_folder_name="trained_models/${model_folder_name}/$folder_prefix"

    if [ -d "$new_folder_name" ]; then
        echo "$new_folder_name already exists.... skipping"
        continue
    fi

    # copying the model from LCC
    echo "$experiment_name - Copying ${wandbids[i]}"
    scp -r ofsk222@lcc.uky.edu:$lcc_baseline_folder ${new_folder_name}

    checkpoint_path=$(find ${new_folder_name} -name "*.ckpt")
    echo "$experiment_name - Training linear probe"
    echo "checkpoint path - $checkpoint_path" 
    echo "$checkpoint_path" > last_ckpt.txt
    
    # # train linear probe
    python3 -u main_linear.py \
        --config-path "scripts/linear/$dataset" \
        --config-name "$method_name" \
        ++name="$experiment_name-linear" \
        ++backbone.name="$backbone" \
    

done
