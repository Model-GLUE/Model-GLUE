## First build merge and moe models
mergekit-yaml /path/to/merge/config /path/to/merge/output --cuda --random-seed 42
mergekit-moe /path/to/mixture/config /path/to/moe/output  --cuda --random-seed 42

export PYTHONPATH="./mixture:$PYTHONPATH"

## when num_dense_layers=0, it is equivalent to a mixture of experts model

## for ffn level (hybrid) mode, run:
python mixture/concat-dense-moe-ffn-level-hybrid.py \
  --dense_dir="/path/to/merge/output" \
  --moe_dir="/path/to/moe/output" \
  --num_dense_layers=16 \
  --out_path="/path/to/final/output" \
  --out_dtype="bfloat16"


## for block/model level (hybrid) run:
##python mixture/concat-dense-moe-model-level-hybrid.py \
# python mixture/concat-dense-moe-block-level-hybrid.py \
#   --expert1_dir="model1" \
#   --expert2_dir="model2" \
#   --more_expert_dir="model3,model4" \
#   --dense_dir="/path/to/merge/output" \
#   --moe_dir="/path/to/moe/output" \
#   --num_dense_layers=16 \
#   --out_path="/path/to/final/output" \
#   --out_dtype="bfloat16"
