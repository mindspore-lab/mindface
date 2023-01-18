## 推理与验证
本文档将介绍如何使用训练好的RetinaFace模型在单张图片做推理，检测出其中所有的人脸。在此之前，请先保证您安装好了相应的环境。

从[此处](https://github.com/mindspore-lab/mindface.git)下载mindface仓库并安装mindface

```shell 
git clone https://github.com/mindspore-lab/mindface.git
cd mindface
python setup.py install
```

成功安装mindspore后，安装依赖包

```shell
cd mindface/detection/
pip install -r requirements.txt
```

## 加载功能包，调用所需函数
在这一部分，我们集中import所需要的功能包，调用之后需要用到的一些函数。

```python
import argparse
import numpy as np
import cv2

from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from utils import prior_box
from models import RetinaFace, resnet50, mobilenet025
from runner import DetectionEngine, read_yaml
```

## 基本设置
选择配置文件为`RetinaFace_mobilenet025.yaml`或者`RetinaFace_resnet50.yaml`，选择`mode`设置为“Graph”即静态图模式，或者设置`mode`为“Pynative”即动态图模式。此处我选择从cfg文件中读取，读者也可自行设置。

```python
#set cfg
config_path = 'mindface/detection/configs/RetinaFace_resnet50.yaml'
cfg = read_yaml(config_path)
#set mode
if cfg['mode'] == 'Graph':
    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
else :
    context.set_context(mode=context.PYNATIVE_MODE, device_target = cfg['device_target'])

```

## 搭建模型
根据配置文件选择`backbone`为MobileNet025或ResNet50，并根据cfg配置文件中给出的路径对应加载验证模型，如果读者自己的checkpoint，可直接添加一行代码修改`cfg['val_model'] = 读者权重路路径`。

```python
#build model
if cfg['name'] == 'ResNet50':
    backbone = resnet50(1001)
elif cfg['name'] == 'MobileNet025':
    backbone = mobilenet025(1000)
network = RetinaFace(phase='predict', backbone=backbone,  in_channel=cfg['in_channel'], out_channel=cfg['out_channel'])
backbone.set_train(False)
network.set_train(False)

#load checkpoint
assert cfg['val_model'] is not None, 'val_model is None.'
param_dict = load_checkpoint(cfg['val_model'])
print('Load trained model done. {}'.format(cfg['val_model']))
network.init_parameters_data()
load_param_into_net(network, param_dict)
```
```text
Load trained model done. /home/user/mindspore/retinaface/retinaface_mindinsight/pretrained/RetinaFace_ResNet50.ckpt
```

## 预设图片尺寸
依据不同需求对图片尺寸进行处理，可选择在原尺寸上进行推理或者裁剪后的尺寸，如果`test_origin_size`为True，则使用原图大小进行推理；否则缩放图片，将其短边和长边尽可能的逼近但不超过1600和2176，缩放结果填充到(2176,2176)大小的画布上面。

```python
# test image
conf_test = cfg['conf']

#choose if you want to infer on origin size or the fixed size
test_origin_size = False

#image_path
image_path = cfg['image_path']

if test_origin_size:
    h_max, w_max = 0, 0

    img_each = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_each.shape[0] > h_max:
        h_max = img_each.shape[0]
    if img_each.shape[1] > w_max:
        w_max = img_each.shape[1]

    h_max = (int(h_max / 32) + 1) * 32
    w_max = (int(w_max / 32) + 1) * 32

    priors = prior_box(image_sizes=(h_max, w_max),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)
else:
    target_size = 1600
    max_size = 2176
    priors = prior_box(image_sizes=(max_size, max_size),
                        min_sizes=[[16, 32], [64, 128], [256, 512]],
                        steps=[8, 16, 32],
                        clip=False)
```

## 检测器初始化
将配置文件中的参数传入runner/engine.py中的DetectionEngine类，对检测器进行初始化，完成之后随即开始推理。

```python
detection = DetectionEngine(nms_thresh = cfg['val_nms_threshold'], conf_thresh = cfg['val_confidence_threshold'], iou_thresh = cfg['val_iou_threshold'], var = cfg['variance'])
```
## 数据预处理
图片先按照预设尺寸进行缩放，后进行归一化并填充维度转成四维张量。

```python
# process the image
img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
img = np.float32(img_raw)
#testing scale
if test_origin_size:
    resize = 1
    assert img.shape[0] <= h_max and img.shape[1] <= w_max
    image_t = np.empty((h_max, w_max, 3), dtype=img.dtype)
    image_t[:, :] = (104.0, 117.0, 123.0)
    image_t[0:img.shape[0], 0:img.shape[1]] = img
    img = image_t
else:
    im_size_min = np.min(img.shape[0:2])
    im_size_max = np.max(img.shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)

    img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    assert img.shape[0] <= max_size and img.shape[1] <= max_size
    image_t = np.empty((max_size, max_size, 3), dtype=img.dtype)
    image_t[:, :] = (104.0, 117.0, 123.0)
    image_t[0:img.shape[0], 0:img.shape[1]] = img
    img = image_t


scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]], dtype=img.dtype)
img -= (104, 117, 123)
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, 0)
img = Tensor(img)
```
## 推理
使用前面初始化完成的检测器进行推理，其中detection类为检测器初始化时实例化的`DetetionEngine`类。

```python
boxes, confs, _ = network(img)
boxes = detection.infer(boxes, confs, resize, scale, priors)
```

其中`infer`函数会输出预测框结果。

```python
def infer(self, boxes, confs, resize, scale, image_path, priors):
    """infer"""
    if boxes.shape[0] == 0:
        # add to result
        event_name, img_name = image_path.split('/')
        self.results[event_name][img_name[:-4]] = {'img_path': image_path,
                                                    'bboxes': []}
        return None

    boxes = decode_bbox(np.squeeze(boxes.asnumpy(), 0), priors, self.var)
    boxes = boxes * scale / resize

    scores = np.squeeze(confs.asnumpy(), 0)[:, 1]
    # ignore low scores
    inds = np.where(scores > self.conf_thresh)[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = self._nms(dets, self.nms_thresh)
    dets = dets[keep, :]

    dets[:, 2:4] = (dets[:, 2:4].astype(np.int32) - dets[:, 0:2].astype(np.int32)).astype(np.float32) # int
    dets[:, 0:4] = dets[:, 0:4].astype(np.int32).astype(np.float32)                                 # int


    # return boxes
    return dets[:, :5].astype(np.float32).tolist()
```

## 结果呈现
将推理完成的图片中目标以锚框选中并标记类别名称，对推理完成的结果图片名称加上后缀以示区别，存放于指定路径中，并将该路径打印呈现.

```python
#show results
img_each = cv2.imread(image_path, cv2.IMREAD_COLOR)

for box in boxes:
    if box[4] > conf_test:
        cv2.rectangle(img_each,(int(box[0]),int(box[1])),
            (int(box[0])+int(box[2]),int(box[1])+int(box[3])),color=(0,0,255))
        cv2.putText(img_each,str(round(box[4],5)),(int(box[0]),int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
save_path = image_path.split('.')[0]+'_pred.jpg'
cv2.imwrite(save_path,img_each)
print(f'Result saving: {save_path}')
```

这部分输出结果为
```text
Result saving: mindface/detection/imgs/0000_pred.jpg
```
就可以看到如图所示的效果了。
![推理结果](/mindface/detection/imgs/0000_pred.jpg)

## 验证
验证逻辑与推理基本相同，区别在于验证时要将所有输出结果拼接起来进行精度评估，并将结果打印出来。

```python
boxes, confs, _ = network(img)
if idx ==0:
    boxes_all = boxes
    confs_all =  confs
    resize_all = [resize]
else:
    boxes_all = ops.concat((boxes_all,boxes))
    confs_all = ops.concat((confs_all,confs),axis=1)
    resize_all.append(resize)
timers['forward_time'].end()

boxes = boxes_all
confs = confs_all
resize = resize_all

timers['misc'].start()
detection.eval(boxes, confs, resize, scale, img_name, priors)
timers['misc'].end()

ave_time = ave_time + timers['forward_time'].diff + timers['misc'].diff
ave_forward_pass_time = ave_forward_pass_time + timers['forward_time'].diff
ave_misc = ave_misc + timers['misc'].diff

forward_time = timers['forward_time'].diff
sum_time = timers['forward_time'].diff + timers['misc'].diff
print(f"im_detect: {i + 1}/{num_images} forward_pass_time: {forward_time:.4f}s",end=' ')
print(f"misc: {timers['misc'].diff:.4f}s sum_time: {sum_time:.4f}s")

print(f"ave_time: {(ave_time/(i+1)):.4f}s")
print(f"ave_forward_pass_time: {(ave_forward_pass_time/(i+1)):.4f}s")
print(f"ave_misc: {(ave_misc/(i+1)):.4f}s")
print('Predict box done.')
print('Eval starting')

# 保存结果至json文件中，其中保存的目录为cfg['val_save_result']
if cfg['val_save_result']:
    # Save the predict result if you want.
    results = detection.write_result(save_path=cfg['val_save_result'])
    assert results is not None, 'Saved Nothing.'
# 计算ap
detection.get_eval_result()
```

输出结果为：
```text
The results were saved in mindface/detection/predict_2023_01_18_14_39_08.json.
Easy   Val Ap : 0.8862
Medium Val Ap : 0.8696
Hard   Val Ap : 0.7993
```

