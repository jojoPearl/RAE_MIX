python stage2_runner.py \
  -i assets/group7/stage1_result/fused_results.pt \
  -o assets/group7/stage2_result \
  --strengths 0.1 0.3 0.4 0.6 0.8 \
  --steps 50


python stage2_runner.py \
  -i assets/group4/stage1_result1/ \
  -o assets/group4/stage2_result1/ \
  --strengths 0.2,0.3,0.4,0.6,0.8,0.9,0.95

python stage2_runner.py \
  -i assets/group4/stage1_result1/ \
  -o assets/group4/stage2_result1/ \
  --seed 123
