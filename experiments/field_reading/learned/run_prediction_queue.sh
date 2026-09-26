#!/bin/zsh
# U4 final GPU queue: full val/eval predictions, then the amount CRNN and its routed predictions.
# Each python job takes and releases the MPS lock itself.
set -u
export HF_HOME=/Volumes/vega/ai-models/field-reading/hf_home TORCH_HOME=/Volumes/vega/ai-models/field-reading/torch_home
PY=/Volumes/vega/datasets/check-transcriber/venv/bin/python
LOGS=/Volumes/vega/datasets/check-transcriber/field-reading/logs
M=experiments.field_reading.learned
LOCALIZATIONS=(oracle segnet_mobilenetv3l_768)
step() { echo "$(date '+%F %T') start $1"; vm_stat | sed -n 2,3p; }
for split in val eval; do for loc in $LOCALIZATIONS; do
  step "crnn_general $split $loc"; $PY -m $M.predict --method crnn_general --split $split --localization $loc > $LOGS/learned_predict__crnn_general__${split}__${loc}.log 2>&1
done; done
for split in val eval; do for loc in $LOCALIZATIONS; do
  step "trocr hw zs $split $loc"; $PY -m $M.predict --method trocr_small_handwritten_zs_grey --split $split --localization $loc > $LOGS/learned_predict__trocr_hw_zs_grey__${split}__${loc}.log 2>&1
done; done
step crnn_amount; $PY -m $M.train_crnn --run-name crnn_amount_h32 --variant crnn_small_h32 --charset amount --fields amount_numeric --epochs 3 --init-from crnn_general_h32 --learning-rate 3e-4 > $LOGS/learned_train_crnn_amount_h32.log 2>&1
for split in val eval; do for loc in $LOCALIZATIONS; do
  step "crnn_amount_route $split $loc"; $PY -m $M.predict --method crnn_amount_route --split $split --localization $loc > $LOGS/learned_predict__crnn_amount_route__${split}__${loc}.log 2>&1
done; done
echo "$(date '+%F %T') queue done"
