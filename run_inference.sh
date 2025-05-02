#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

cd SadTalker

# Run inference with parameters
python3.8 inference.py \
  --driven_audio "$1" \
  --source_image "$2" \
  --result_dir "$3" \
  --still --preprocess full \
  --batch_size 1 \
  --size 256

# Deactivate virtual environment
deactivate