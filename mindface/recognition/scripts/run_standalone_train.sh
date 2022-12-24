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
echo "For example: bash run.sh path/MS1M DEVICE_ID"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

export CONFIG=$1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

python train.py --device_target 'Ascend' --config $1 --batch_size 64

