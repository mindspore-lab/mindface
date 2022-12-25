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
echo "bash run.sh DATA_PATH RANK_SIZE"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e

export RANK_SIZE=$2
export CONFIG=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=2,3
# export CUDA_VISIBLE_DEVICES=4,5
# export CUDA_VISIBLE_DEVICES=6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3

env > env_distribute_gpu.log
echo "start training"
mpirun -n $2 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py --device_target 'GPU' --config $1 --batch_size 128