#!/bin/bash

# List of all camera bench splits
SPLITS=(
  "fixed_camera"
  "has_backward_wrt_camera"
  "has_downward_wrt_camera"
  "has_forward_wrt_camera"
  "has_leftward"
  "has_pan_left"
  "has_pan_right"
  "has_rightward"
  "has_roll_clockwise"
  "has_roll_counterclockwise"
  "has_tilt_down"
  "has_tilt_up"
  "has_upward_wrt_camera"
  "has_zoom_in"
  "has_zoom_out"
)

# Models to evaluate
MODELS=("qwen2.5-vl" "qwen2.5-vl-ft")
DATA="sugarcrepe"
TRAIN_DIR="data/camerabench_trainset"
TEST_DIR="data/camerabench_testset"

# Create logs directories for each model
mkdir -p logs/qwen2.5-vl
mkdir -p logs/qwen2.5-vl-ft

# Run for each model
for MODEL in "${MODELS[@]}"; do
  echo "Starting evaluations for model: $MODEL"
  
  # Create model-specific log directory
  LOG_DIR="logs/$MODEL"
  
  # GPUs 0,1 - Run splits 0-3 sequentially
  (export CUDA_VISIBLE_DEVICES=0,1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[0]}.jsonl" --val_path "$TEST_DIR/${SPLITS[0]}.jsonl" > "$LOG_DIR/${SPLITS[0]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[1]}.jsonl" --val_path "$TEST_DIR/${SPLITS[1]}.jsonl" > "$LOG_DIR/${SPLITS[1]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[2]}.jsonl" --val_path "$TEST_DIR/${SPLITS[2]}.jsonl" > "$LOG_DIR/${SPLITS[2]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[3]}.jsonl" --val_path "$TEST_DIR/${SPLITS[3]}.jsonl" > "$LOG_DIR/${SPLITS[3]}.log" 2>&1) &

  # GPUs 2,3 - Run splits 4-6 sequentially
  (export CUDA_VISIBLE_DEVICES=2,3; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[4]}.jsonl" --val_path "$TEST_DIR/${SPLITS[4]}.jsonl" > "$LOG_DIR/${SPLITS[4]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[5]}.jsonl" --val_path "$TEST_DIR/${SPLITS[5]}.jsonl" > "$LOG_DIR/${SPLITS[5]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[6]}.jsonl" --val_path "$TEST_DIR/${SPLITS[6]}.jsonl" > "$LOG_DIR/${SPLITS[6]}.log" 2>&1) &
    
  # GPUs 4,5 - Run splits 7-9 sequentially
  (export CUDA_VISIBLE_DEVICES=4,5; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[7]}.jsonl" --val_path "$TEST_DIR/${SPLITS[7]}.jsonl" > "$LOG_DIR/${SPLITS[7]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[8]}.jsonl" --val_path "$TEST_DIR/${SPLITS[8]}.jsonl" > "$LOG_DIR/${SPLITS[8]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[9]}.jsonl" --val_path "$TEST_DIR/${SPLITS[9]}.jsonl" > "$LOG_DIR/${SPLITS[9]}.log" 2>&1) &
    
  # GPUs 6,7 - Run splits 10-14 sequentially
  (export CUDA_VISIBLE_DEVICES=6,7; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[10]}.jsonl" --val_path "$TEST_DIR/${SPLITS[10]}.jsonl" > "$LOG_DIR/${SPLITS[10]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[11]}.jsonl" --val_path "$TEST_DIR/${SPLITS[11]}.jsonl" > "$LOG_DIR/${SPLITS[11]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[12]}.jsonl" --val_path "$TEST_DIR/${SPLITS[12]}.jsonl" > "$LOG_DIR/${SPLITS[12]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[13]}.jsonl" --val_path "$TEST_DIR/${SPLITS[13]}.jsonl" > "$LOG_DIR/${SPLITS[13]}.log" 2>&1; \
   python -m src.run_camera_bench --model_name "$MODEL" --data_name "$DATA" --train_path "$TRAIN_DIR/${SPLITS[14]}.jsonl" --val_path "$TEST_DIR/${SPLITS[14]}.jsonl" > "$LOG_DIR/${SPLITS[14]}.log" 2>&1) &

  # Wait for all jobs for this model to complete
  wait
  
  # Print a summary for this model
  echo "Evaluations complete for model: $MODEL"
  echo "Results for $MODEL:"
  for split in "${SPLITS[@]}"; do
    echo -n "$split: "
    grep "Raw Accuracy" "$LOG_DIR/$split.log" | tail -1
  done
  echo ""
done

# Final comparison
echo "Comparison of models across all splits:"
echo "-----------------------------------------"
printf "%-25s %-15s %-15s\n" "Split" "qwen2.5-vl" "qwen2.5-vl-ft"
echo "-----------------------------------------"
for split in "${SPLITS[@]}"; do
  base_acc=$(grep "Raw Accuracy" "logs/qwen2.5-vl/$split.log" | tail -1 | awk '{print $3}')
  ft_acc=$(grep "Raw Accuracy" "logs/qwen2.5-vl-ft/$split.log" | tail -1 | awk '{print $3}')
  printf "%-25s %-15s %-15s\n" "$split" "$base_acc" "$ft_acc"
done
echo "-----------------------------------------"