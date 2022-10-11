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

# extract images
python src/rec2jpg_dataset.py --include /home/data/xieguochen/dataset/AgeDataset/faces_webface_112x112 \
 --output /home/data/xieguochen/dataset/AgeDataset/faces_webface_112x112_train

# MS1M V2
# distributed
sh scripts/run_distribute_train_gpu.sh  configs/train_config_ms1m.yaml 2
# single
sh scripts/run_standalone_train_gpu.sh  configs/train_config_ms1m.yaml

# CASIA
# distributed
sh scripts/run_distribute_train_gpu.sh  configs/train_config_casia.yaml 2
# single
sh scripts/run_standalone_train_gpu.sh  configs/train_config_casia.yaml

# Eval
sh scripts/run_eval_gpu.sh /home/data/dushens/dataset/mindspore/faces_emore /home/data/xieguochen/code/mindspore_resnet/train_parallel_r50_casia/ArcFace--25_958.ckpt iresnet50

