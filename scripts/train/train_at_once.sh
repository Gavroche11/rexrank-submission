#!/bin/bash

DEVICES="0,1"

bash scripts/train/pretrain.sh "$DEVICES" && \
bash scripts/train/finetune.sh "$DEVICES"