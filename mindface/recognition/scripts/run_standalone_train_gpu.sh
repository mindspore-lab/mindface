#!/bin/bash

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh DATA_PATH"
echo "For example: bash run.sh /path/dataset"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

export CONFIG=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=0
export DEVICE_ID=0

rm -rf ./train_single
mkdir ./train_single

cp -r ./configs/ ./train_single
cp -r ./datasets/ ./train_single
cp -r ./loss/ ./train_single
cp -r ./models/ ./train_single
cp -r ./scripts/ ./train_single
cp -r ./test/ ./train_single
cp -r ./utils/ ./train_single
# shellcheck disable=SC2035
cp *.py ./train_single

cd ./train_single
env > env.log
echo "start training"

python train.py --device_target 'GPU' --device_num 1 --config $1 
# > train.log 2>&1 &
