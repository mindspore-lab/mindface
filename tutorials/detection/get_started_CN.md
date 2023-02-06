## 快速开始 

### 教程

```python
>>> import mindface 
>>> from models import resnet50, mobilenet025
# 创建骨干网络
>>> backbone = mobilenet025(1001)
# 创建模型
>>> network = RetinaFace(phase='train', backbone=backbone, in_channel=in_channel, out_channel=out_channel)
# 验证精度
>>> !python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
{'Easy  Val AP': 0.9446, 'Medium Val AP': 0.9364, 'Hard  Val AP': 0.8942}
```

**推理图像示例**

使用预训练模型对输入的图像进行推理，

```python
>>> !python infer.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml \        
        --checkpoint pretrained/weight.ckpt --image_path test/detection/imgs/0000.jpg --conf 0.5
```

使用我们预训练的权重进行推理的结果如下图所示.
![retinaface_picture](/test/detection/imgs/0000_pred.jpg)

### 脚本指南
用MindFace在标准数据集或你自己的数据集上训练你的模型很容易。模型训练、转移学习或评估可以通过灵活的配置使用一行或几行代码完成。

- 训练

    使用`train.py`很容易训练你的模型，其中的训练策略（例如，增强，LR调度）可以用外部参数或yaml配置文件来配置。

    - 标准训练
    ```shell
        python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```

    - 分布式训练

        要在分布式模式下运行，需要安装[openmpi]（https://www.open-mpi.org/software/ompi/v4.0/）。 

    ```shell
        export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs
        mpirun -n 4 python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```

    >  注意：如果你的设备是Ascend，请将配置文件中的 "device_target "设置为 "Ascend"。





- 验证

    用`eval.py`来验证一个训练好的模型是很容易的。
    ```python
    # 对训练好的checkpoint进行验证
    python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
    ``` 
