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
        --checkpoint pretrained/weight.ckpt --image_path ./imgs/0000.jpg --conf 0.5
```

### Useful Script Guidelines
It is easy to train your model on standard datasets or your own dataset with MindFace. Model training, transfer learning, or evaluaiton can be done using one or a few line of code with flexible configuration. 

- Standalone Training

It is easy to do model training with `train.py`. Here is an example for training a RetinaFace with mobilenet on WiderFace dataset using one computing device (i.e., standalone GPU).
``` shell
python train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml
```



**Resume training.** To resume training, please specify `--resume_net` for the checkpoint. The optimizer state including learning rate of the last epoch will also be recovered. 

```python
python train.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml \
		--resume_net=checkpoints
``` 




- Validation

It is easy to validate a trained model with `eval.py`. 
```python
# validate a trained checkpoint
python eval.py --config mindface/detection/configs/RetinaFace_mobilenet025.yaml --checkpoint pretrained/weight.ckpt
``` 
