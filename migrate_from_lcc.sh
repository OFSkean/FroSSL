#tradeoffs=(0.4 0.5 0.6 0.7 0.8 0.9 1.0)

wandbids=("ivne/3mopec3b") # "corinfomax/hijw42t3" "corinfomax/pm2us0et")
#wandbids=("dino/urt3o574" "corinfomax/axxtovsi" "searmse/wy0vxvcg" "barlow/4049ryr3")
model_folder_name="ivne"
dataset="cifar100"
backbone="resnet18"
mkdir trained_models/sweeps/${model_folder_name}

# check if data is cifar
if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
    dataset_config_name="cifar"
else
    dataset_config_name="$dataset"
fi


arraylength=${#wandbids[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    method_name=$(echo ${wandbids[i]} |cut -d/ -f 1)
    wandbid=$(echo ${wandbids[i]} |cut -d/ -f 2)
    experiment_name="$method_name-$dataset-$wandbid"

    folder_prefix="$experiment_name"
    lcc_baseline_folder="/home/ofsk222/projects/frossl/solo-learn/trained_models/${wandbids[i]}"
    chunky_baseline_folder="/home/oskean/Research/frossl/solo-learn/trained_models/${wandbids[i]}"
    new_folder_name="trained_models/sweeps/${model_folder_name}/$folder_prefix"

    # if [ -d "$new_folder_name" ]; then
    #     echo "$new_folder_name already exists.... skipping"
    #     continue
    # fi

    # # # copying the model from LCC
    echo "$experiment_name - Copying ${wandbids[i]}"
    echo "ofsk222@lcc.uky.edu:$chunky_baseline_folder"
    #scp -r oskean@10.163.149.137:$chunky_baseline_folder ${new_folder_name}
    scp -r ofsk222@lcc.uky.edu:$lcc_baseline_folder ${new_folder_name}

    checkpoint_path=$(find ${new_folder_name} -name "*.ckpt")
    echo "$experiment_name - Training linear probe"
    echo "checkpoint path - $checkpoint_path"
    #echo "$checkpoint_path" > last_ckpt.txt 
    echo $checkpoint_path > last_ckpt.txt
    sleep 1
    # # train linear probe
    python3 -u main_linear.py \
        --config-path "scripts/linear/$dataset_config_name" \
        --config-name "$method_name" \
        ++name="$experiment_name-linear-$wandbid" \
        ++backbone.name="$backbone" \
        ++data.dataset="$dataset" \
        ++auto_resume.enabled=False

done
