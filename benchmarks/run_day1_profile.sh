#!/usr/bin/env bash
set -euo pipefail

variant=${1:?usage: run_day1_profile.sh VARIANT [BATCH] [OUTPUT_TOKENS] [PROMPT_TOKENS]}
batch=${2:-16}
output_tokens=${3:-128}
prompt_tokens=${4:-128}

project_dir=${PROJECT_DIR:-/data/rag/pluo35/nano-vllm-issue175}
model_dir=${MODEL_DIR:-/data/rag/pluo35/models/Qwen3-0.6B}
result_dir=${RESULT_DIR:-$project_dir/results/day1/$variant/batch-$batch}
python_bin=${PYTHON_BIN:-$project_dir/.venv/bin/python}

cd "$project_dir"
mkdir -p "$result_dir"

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to the physical project GPU}"
export HF_HOME=${HF_HOME:-/data/rag/pluo35/hf_cache}
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
unset TRANSFORMERS_CACHE || true

nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,compute_mode \
  --format=csv > "$result_dir/gpu-before.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader > "$result_dir/gpu-processes-before.csv" || true

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$result_dir/$variant" \
  "$python_bin" benchmarks/benchmark_decode_hotpath.py \
    --model "$model_dir" \
    --output-dir "$result_dir" \
    --variant "$variant" \
    --batch-size "$batch" \
    --prompt-tokens "$prompt_tokens" \
    --output-tokens "$output_tokens" \
    --warmup-output-tokens 32 \
    --measured-runs 3 \
    --cuda-profiler-range \
    --nvtx

nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum \
  --format csv "$result_dir/$variant.nsys-rep" \
  > "$result_dir/$variant.nsys-stats.csv"

"$python_bin" benchmarks/analyze_decode_profiles.py \
  "$result_dir/$variant.steps.jsonl" \
  --output "$result_dir/$variant.step-summary.json"

nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu,compute_mode \
  --format=csv > "$result_dir/gpu-after.csv"
