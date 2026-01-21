python stage1_runner.py \
  --base assets/group4/base.png \
  --objects assets/group4/objects.json \
  --outdir assets/group4/stage1_out \
  --fused_prefix assets/group4/stage1_out/fused_results.pt \
  --global_scales 0.3,0.35,0.4,0.45,0.5 \
  --fusion_mode v2 \
  --rerank_clip 1 \
  --overlap_mode no_overwrite


python stage1_runner.py \
  --base assets/group4/base.png \
  --objects assets/group4/objects.json \
  --outdir assets/group4/stage1_out_m2 \
  --fused_prefix assets/group4/stage1_out_m2/fused_results.pt \
  --global_scales 0.2,0.4,0.6,0.8,1.2 \
  --fusion_mode m2 \
  --rerank_clip 0 \
  --overlap_mode allow
