
import argparse
import pytest
from src.dataset import create_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test_module')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='train batchsize (default: 256)')
    parser.add_argument('--target', type=str, default='GPU', choices=['GPU', 'Ascend'])
    parser.add_argument('--data_url', default='data path', type=str)
    args = parser.parse_args()
    
    train_dataset = create_dataset(dataset_path=args.data_url, do_train=True,
                                       repeat_num=1, batch_size=args.batch_size, target=args.target)
    assert train_dataset.get_batch_size() == args.batch_size
