
import argparse
import pytest
from datasets.face_dataset import create_dataset



data_url = 'path to your dataset'
batch_size = 256
target = 'GPU'
train_dataset = create_dataset(dataset_path=data_url, do_train=True,
                                    repeat_num=1, batch_size=batch_size, target=target)
assert train_dataset.get_batch_size() == batch_size
