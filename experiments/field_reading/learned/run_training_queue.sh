#!/bin/zsh
# Sequential U4 MPS jobs; each python job takes and releases the MPS lock itself.
# Launch only when the lead has released the GPU to U4. Optional $1: a PID to wait for first.
# Usage: nohup experiments/field_reading/learned/run_training_queue.sh [pid] > <log> 2>&1 &
set -u
export HF_HOME=/Volumes/vega/ai-models/field-reading/hf_home TORCH_HOME=/Volumes/vega/ai-models/field-reading/torch_home
PY=/Volumes/vega/datasets/check-transcriber/venv/bin/python
LOGS=/Volumes/vega/datasets/check-transcriber/field-reading/logs
M=experiments.field_reading.learned
[[ $# -ge 1 ]] && while kill -0 "$1" 2>/dev/null; do sleep 30; done
step() { echo "$(date '+%F %T') start $1"; vm_stat | sed -n 2,3p; }
step crnn_general;  $PY -m $M.train_crnn --run-name crnn_general_h32 --variant crnn_small_h32 --charset general --epochs 5 > $LOGS/learned_train_crnn_general_h32.log 2>&1
step crnn_predict_val; $PY -m $M.predict --method crnn_general --split val > $LOGS/learned_predict_crnn_general_val.log 2>&1
for method in trocr_small_printed_zs trocr_small_handwritten_zs trocr_small_handwritten_zs_grey; do
  step "screen $method"; $PY -m $M.predict --method $method --split val --rows-per-field 300 > $LOGS/learned_screen_$method.log 2>&1
done
step finetune_trocr; $PY -m $M.finetune_trocr --max-minutes 90 > $LOGS/learned_finetune_trocr.log 2>&1
step crnn_amount;   $PY -m $M.train_crnn --run-name crnn_amount_h32 --variant crnn_small_h32 --charset amount --fields amount_numeric --epochs 3 --init-from crnn_general_h32 --learning-rate 3e-4 > $LOGS/learned_train_crnn_amount_h32.log 2>&1
echo "$(date '+%F %T') queue done"
