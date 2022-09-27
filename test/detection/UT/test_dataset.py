# import packages
import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(),'mindface/detection'))

from mindface.detection.datasets import create_dataset

def test_dataset(cfg):
    batch_size = 8
    data_dir = cfg['training_dataset']
    ds_train = create_dataset(data_dir, cfg, batch_size, multiprocessing=True, num_worker=2)
    assert ds_train.get_batch_size() == batch_size


