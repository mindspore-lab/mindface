"""
Inference of face recognition models.
"""
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

import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .models import iresnet100, iresnet50, get_mbf

def infer(img, backbone="iresnet50", num_features=512, pretrained=False):
    """
    The inference of arcface.

    Args:
        img (NumPy): The input image.
        backbone (Object): Arcface model without loss function. Default: "iresnet50".
        pretrained (Bool): Pretrain. Default: False.

    Examples:
        >>> img = input_img
        >>> out1 = infer(input_img, backbone="iresnet50",
                        pretrained="/path/to/eval/ArcFace.ckpt")
    """
    assert (img.shape[-1] == 112 and img.shape[-2] == 112)
    img = ((img / 255) - 0.5) / 0.5
    img = ms.Tensor(img, ms.float32)
    if len(img.shape) == 4:
        pass
    elif len(img.shape) == 3:
        img = img.expand_dims(axis=0)

    if backbone == "iresnet50":
        model = iresnet50(num_features=num_features)
    elif backbone == "iresnet100":
        model = iresnet100(num_features=num_features)
    elif backbone == "mobilefacenet":
        model = get_mbf(num_features=num_features)
    else:
        raise NotImplementedError

    if pretrained:
        param_dict = load_checkpoint(pretrained)
        load_param_into_net(model, param_dict)

    net_out = model(img)
    embeddings = net_out.asnumpy()

    return embeddings
