python3 ../../../main_pretrain.py \
    --dataset imagenet100 \
    --encoder resnet18 \
    --data_dir /data/datasets \
    --train_dir imagenet-100/train \
    --val_dir imagenet-100/val \
    --max_epochs 400 \
    --gpus 0,1 \
    --accelerator ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm \
    --scheduler warmup_cosine \
    --lr 0.5 \
    --classifier_lr 0.1 \
    --weight_decay 1e-5 \
    --batch_size 128 \
    --num_workers 4 \
    --dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --name byol-400ep-imagenet100 \
    --entity unitn-mhug \
    --project solo-learn \
    --wandb \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 8192 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0
