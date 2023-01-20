# 模型训练/微调

MindFace 套件支持用户加载配置文件中进行设置加载预训练的模型权重，在自身任务上进行微调（finetune）。

## 启动训练/微调

训练和微调的步骤基本相同。如果进行微调或加载预训练权重，请在模型配置文件中的修改`resume`参数，并设置权重路径，示例如下：

```python
resume: '/path/to/ckpt'
```

其余训练步骤不变，代码将在训练过程中完成模型的加载及后续训练流程。

启动模型训练的方式如下：

### 单卡训练/微调

GPU平台

```python
python train.py --config 'configs/train_config_casia_vit_t.yaml' --device_target 'GPU'
```

Ascend平台

```python
python train.py --config 'configs/train_config_casia_vit_t.yaml' --device_target 'Ascend'
```

### 分布式训练/微调

GPU平台

```python
mpirun -n 4 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py --config 'configs/train_config_casia_vit_t.yaml' --device_target 'GPU'
```

Ascend平台

```python
mpirun -n 4 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
python train.py --config 'configs/train_config_casia_vit_t.yaml' --device_target 'Ascend'
```
