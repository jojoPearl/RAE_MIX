python stage1_runner.py \
  --base assets/group4/base.png \
  --objects assets/group4/objects.json \
  --out assets/group4/stage1_result_basic \
  --overlap_mode allow \
  --gscales 0.3 0.35 0.4 0.45 0.5


python stage1_runner.py \
  --base assets/group4/base.png \
  --objects assets/group4/objects.json \
  --out assets/group4/stage1_result_clip \
  --overlap_mode no_overwrite \
  --rerank_clip \
  --clip_model openai/clip-vit-base-patch32
