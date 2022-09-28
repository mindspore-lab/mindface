import argparse
# import pytest
from mindface.recognition.datasets import create_dataset
import sys

import pytest

MAX = 256 # 160146
@pytest.mark.parametrize('batch_size', [1, MAX])
@pytest.mark.parametrize('target', ['GPU', 'Ascend'])
def test_create_dataset(batch_size, target):
    data_url = '/home/d1/xieguochen/dataset/AgeDatasets/faces_webface_112x112_train'
    train_dataset = create_dataset(data_url, do_train=True,
                                       repeat_num=1, batch_size=batch_size, target=target, is_parallel=False)
    
    assert train_dataset.get_batch_size() == batch_size, 'create_dataset test failed'
