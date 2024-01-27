# python3 main_pretrain.py \
#     --config-path scripts/pretrain/cifar-multicrop \
#     --config-name swav.yaml \
#     ++name="swav-cifar10-multicrop" \

python3 main_linear.py \
    --config-path scripts/linear/imagenet100 \
    --config-name frossl.yaml \
    ++name="frossl-2view-in100-0.9" \
