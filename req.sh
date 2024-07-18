#!/bin/bash

echo "start installing dependencies..."

pip install einops
pip install kornia
pip install timm==0.6.13
pip install causal-conv1d==1.1.0
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.1/index.html
apt-get update -y
apt-get install libglib2.0-0 -y

echo "Finished"
