#!/bin/bash

DEVICE="0"

for dataset in mimic chexpert openi; do
    bash scripts/eval/eval_per_dataset.sh "$dataset" "$DEVICE" &
done
wait