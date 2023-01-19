## Get Started 

### Hands-on Tutorial

```python
>>> import mindface 
>>> from models import resnet50, mobilenet025
# Create the backbone
>>> backbone = mobilenet025(1001)
# Create the model object
>>> network = RetinaFace(phase='train', backbone=backbone, in_channel=in_channel, out_channel=out_channel)
# Validate its accuracy
>>> !python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
{'Easy  Val AP': 0.9446, 'Medium Val AP': 0.9364, 'Hard  Val AP': 0.8942}
```

**Image infer demo**

Infer the input image with a pretrained SoTA model,

```python
>>> !python infer.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml \        
        --checkpoint pretrained/weight.ckpt --image_path test/detection/imgs/0000.jpg --conf 0.5
```

The inference results by using our pre-trained weights are shown in the following figure.
![retinaface_picture](/test/detection/imgs/0000_pred.jpg)

### Useful Script Guidelines
It is easy to train your model on standard datasets or your own dataset with MindFace. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. 

- Training

    It is easy to train your model using `train.py`, where the training strategy (e.g., augmentation, LR scheduling) can be configured with external arguments or a yaml config file.

    - Standalone Training
    ```shell
        python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```

    - Distributed Training

        To run in distributed mode, [openmpi](https://www.open-mpi.org/software/ompi/v4.0/) is required to install.  

    ```shell
        export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs
        mpirun -n 4 python mindface/detection/train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
    ```

    > note: if your device is Ascend, please set the "device_target" in the config file to "Ascend".





- Validation

    It is easy to validate a trained model with `eval.py`. 
    ```python
    # validate a trained checkpoint
    python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
    ``` 
