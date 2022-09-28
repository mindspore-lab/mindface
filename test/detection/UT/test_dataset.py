# import packages
from mindface.detection.datasets import create_dataset

def test_dataset(cfg):
    batch_size = 8
    data_dir = cfg['training_dataset']
    ds_train = create_dataset(data_dir, cfg, batch_size, multiprocessing=True, num_worker=2)
    assert ds_train.get_batch_size() == batch_size


