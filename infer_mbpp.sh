model= # your model path
output_path= # define the output path
index=0
gpu_num=8
temp=0.8
max_len=2048
pred_num=10
num_seqs_per_iter=10

# run with 8 gpu
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 63))
  end_index=$(((i + 1) * 63))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python3 src/mbpp_gen.py \
      --model ${model} \
      --start_index ${start_index} \
      --end_index ${end_index} \
      --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} \
      --N ${pred_num} \
      --max_len ${max_len} \
      --output_path ${output_path} \
      --mbpp_path ${mbpp_path} \
      --prompt_type origin \
      2>&1 | tee $output_path/log/$((i)).log
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done