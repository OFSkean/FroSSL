tradeoffs=(0.4 0.5 0.6 0.7 0.8 0.9)
wandbids=("46s9gri0"  "3id6g480" "5xcmrkqt" "zro7mcf5" "ay26etu2" "c7iawwla")
model_folder_name="8view_constant_sweep"
folder_prefix="constant-"

mkdir trained_models/${model_folder_name}

arraylength=${#tradeoffs[@]}

for (( i=0; i<${arraylength}; i++ ));
do
    experiment_name="frossl-4view-constant-${tradeoffs[i]}"

    lcc_baseline_folder="/home/ofsk222/projects/frossl/solo-learn/trained_models/searmse/${wandbids[i]}"
    new_folder_name="trained_models/${model_folder_name}/$folder_prefix${tradeoffs[i]}"

    echo "$experiment_name - Copying tradeoff ${tradeoffs[i]} with wandb id ${wandbids[i]}"
    scp -r ofsk222@lcc.uky.edu:$lcc_baseline_folder ${new_folder_name}

    checkpoint_path=$(find ${new_folder_name} -name "fro*")
    echo "$experiment_name - Training linear probe"
    echo "checkpoint path - $checkpoint_path"
    ./linear.sh $experiment_name ${tradeoffs[i]} $checkpoint_path

done