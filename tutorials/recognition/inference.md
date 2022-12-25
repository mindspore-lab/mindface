# 模型验证

MindFace支持人脸识别模型在多个验证集上验证模型性能，当前支持的人脸验证数据集包括lfw,cfp_fp,agedb_30,calfw,cplfw

## 验证数据集

MindFace人脸验证数据集使用.bin格式，所有的验证集全部放在同一目录下，数据集的组织结构如下所示：

```
val_dataset
 ├── agedb_30.bin
 ├── calfw.bin
 ├── cfp_fp.bin
 ├── cplfw.bin
 └── lfw.bin
```

### 执行验证程序

GPU平台

```
sh scripts/run_eval_gpu.sh /path/to/val_dataset /path/to/ckpt model_name
```

Ascend平台

```
sh scripts/run_eval.sh /path/to/val_dataset /path/to/ckpt model_name
```

等待模型验证完成即可得到在验证集上的验证结果：
```
Finish loading vit_b
[WARNING] ME(2133442:139775108977472,MainProcess):2022-12-17-17:20:09.919.747 [mindspore/train/serialization.py:734] For 'load_param_into_net', remove parameter prefix name: _backbone., continue to load.
model loading time 14.051478
['lfw', 'cfp_fp', 'agedb_30', 'calfw', 'cplfw']
loading..  lfw
loading..  cfp_fp
loading..  agedb_30
loading..  calfw
loading..  cplfw
testing verification..
(12000, 512)
infer time 86.12441499999998
[lfw]XNorm: 19.466167
[lfw]Accuracy: 0.99817+-0.00252
[lfw]Accuracy-Flip: 0.99750+-0.00327
testing verification..
(14000, 512)
infer time 96.53912000000001
[cfp_fp]XNorm: 20.283825
[cfp_fp]Accuracy: 0.94200+-0.01296
[cfp_fp]Accuracy-Flip: 0.94671+-0.00845
testing verification..
(12000, 512)
infer time 81.65473200000004
[agedb_30]XNorm: 20.725768
[agedb_30]Accuracy: 0.97517+-0.00858
[agedb_30]Accuracy-Flip: 0.97600+-0.00863
testing verification..
(12000, 512)
infer time 79.85072499999993
[calfw]XNorm: 19.212714
[calfw]Accuracy: 0.96000+-0.01179
[calfw]Accuracy-Flip: 0.96017+-0.01230
testing verification..
(12000, 512)
infer time 76.116541
[cplfw]XNorm: 20.787293
[cplfw]Accuracy: 0.90967+-0.01152
[cplfw]Accuracy-Flip: 0.91317+-0.01102
```
# 部署
## 安装
MindSpore Serving当前仅支持Linux环境部署。

MindSpore Serving包在各类硬件平台（Nvidia GPU, Ascend 910/310P/310, CPU）上通用，推理任务依赖MindSpore或MindSpore Lite推理框架，我们需要选择一个作为Serving推理后端。当这两个推理后端同时存在的时候，优先使用MindSpore Lite推理框架。
具体安装的步骤可以查看[官方的教程](https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_zh_cn/serving_install.md#https://gitee.com/mindspore/docs/blob/master/docs/serving/docs/source_zh_cn/serving_install.md)

## 模型导出
在模型训练完后，训练完成后的网络模型（即CKPT文件）转换为MindIR格式，用于后续手机侧的推理。通过`export`接口会在当前目录下会生成`xxxxx.mindir`文件。

```python
import mindspore as ms
from mindface.recognition.models import *

# 定义并加载网络参数
model_name = "iresnet50"

if model_name == "iresnet50":
    model = iresnet50(num_features=512)
elif model_name == "iresnet100":
    model = iresnet100(num_features=512)
elif model_name == "mobilefacenet":
    model = get_mbf(num_features=512)
elif train_info['backbone'] == 'vit_t':
    net = vit_t(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_s':
    net = vit_s(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_b':
    net = vit_b(num_features=train_info['num_features'])
elif train_info['backbone'] == 'vit_l':
    net = vit_l(num_features=train_info['num_features'])
else:
    raise NotImplementedError

param_dict = ms.load_checkpoint("iresnet50.ckpt")
ms.load_param_into_net(net, param_dict)

# 将模型由ckpt格式导出为MINDIR格式
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 112, 112]).astype(np.float32)
ms.export(net, ms.Tensor(input_np), file_name="iresnet50_112", file_format="MINDIR")
```
导出成功后，会产生一`iresnet50_112.mindir`结尾的文件。

## 推理目录结构介绍
创建目录放置推理代码工程，例如`/home/HwHiAiUser/Ascend/ascend-toolkit/20.0.RC1/acllib_linux.arm64/sample/acl_execute_model/ascend310_resnet50_preprocess_sample`，可以从[官网示例](https://gitee.com/mindspore/docs/tree/r1.8/docs/sample_code/ascend310_resnet50_preprocess_sample)下载样例代码，`model`目录用于存放上述导出的`MindIR`模型文件，`test_data`目录用于存放待识别的图片，推理代码工程目录结构如下:

```
└─ascend310_resnet50_preprocess_sample
    ├── CMakeLists.txt                           // 构建脚本
    ├── README.md                                // 使用说明
    ├── main.cc                                  // 主函数1，手动定义预处理的模型推理方式
    ├── main_hide_preprocess.cc                  // 主函数2，免预处理代码的推理方式
    ├── model
    │   ├── iresnet50_112.mindir                 // MindIR模型文件
    └── test_data
        ├── test1.png                            // 输入样本图片1
        ├── test2.png                            // 输入样本图片2
        ├── ...                                  // 输入样本图片n

```
## 部署Serving推理服务
### 配置
```
demo
├── iresenet50_112
│   ├── 1
│   │   └── iresenet50_112.mindir
│   └── servable_config.py
│── serving_server.py
├── serving_client.py

```
其中，模型配置文件`serving_config.py`内容如下：
```python
import numpy as np
from mindspore_serving.server import register


def add_trans_datatype(x1, x2):
    """define preprocess, this example has two inputs and two outputs"""
    return x1.astype(np.float32), x2.astype(np.float32)

# when with_batch_dim is set to False, only 2x2 add is supported
# when with_batch_dim is set to True(default), Nx2 add is supported, while N is viewed as batch
# float32 inputs/outputs
model = register.declare_model(model_file="tensor_add.mindir", model_format="MindIR", with_batch_dim=False)

# register add_common method in add
@register.register_method(output_names=["y"])
def add_common(x1, x2):  # only support float32 inputs
    """method add_common data flow definition, only call model"""
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y

# register add_cast method in add
@register.register_method(output_names=["y"])
def add_cast(x1, x2):
    """method add_cast data flow definition, only preprocessing and call model"""
    x1, x2 = register.add_stage(add_trans_datatype, x1, x2, outputs_count=2)  # cast input to float32
    y = register.add_stage(model, x1, x2, outputs_count=1)
    return y

```
### 启动服务
Mindspore的server函数提供两种服务部署，本教程以gRPC方式为例。另一种请查看[官方教程](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html)

执行`serving_server.py`，完成服务启动。
```python
import os
import sys
from mindspore_serving import server


def start():
    servable_dir = os.path.dirname(os.path.realpath(sys.argv[0]))

    servable_config = server.ServableStartConfig(servable_directory=servable_dir, servable_name="add",
                                                 device_ids=(0, 1))
    server.start_servables(servable_configs=servable_config)

    server.start_grpc_server(address="127.0.0.1:5500")
    server.start_restful_server(address="127.0.0.1:1500")


if __name__ == "__main__":
    start()

```
在shell中打印出下列的日志，说明启动成功。
```
Serving gRPC server start success, listening on 127.0.0.1:5500
```

### 执行推理
本文使用gRPC的方式为例。使用`serving_client.py`，启动客户端，即可得到模型推理结果。
