python3 main_pretrain.py \
    --config-path scripts/pretrain/stl10 \
    --config-name empssl.yaml \
    ++name="empssl-stl10-20" \

python3 main_linear.py \
    --config-path scripts/linear/stl10 \
    --config-name empssl.yaml \
    ++name="empssl-stl10-20-linear" \
