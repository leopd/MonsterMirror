#!/bin/bash

set -e
set -x
cd pretrained-models

echo "Downloading face detector"
wget https://github.com/leopd/SFD_pytorch/releases/download/0.3/s3fd_model.tgz
tar xvzf s3fd_model.tgz
rm s3fd_model.tgz

echo "Downloading GANimal model"
pip install gdown
gdown https://drive.google.com/uc?id=1CsmSSWyMngtOLUL5lI-sEHVWc2gdJpF9
tar xvzf pretrained.tar.gz
rm pretrained.tar.gz
rm animal119_gen_00100000.pt  # we use the other one.
