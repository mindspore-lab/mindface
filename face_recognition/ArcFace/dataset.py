'''
Dataset: 该部分负责引入训练、验证、测试所用的数据集
'''

import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size


class MsFaceDataset():
    def __init__(self, dataset_dir, target="Ascend"):
        super(MsFaceDataset, self).__init__()
        if target == "Ascend":
            device_num, rank_id = _get_rank_info()
        else:
            init("nccl")
            device_num = get_group_size()
            rank_id = get_rank()

        if device_num == 1:
            self.dataset = de.ImageFolderDataset(
                dataset_dir, num_parallel_workers=8, shuffle=True)
        else:
            self.dataset = de.ImageFolderDataset(
                dataset_dir, num_parallel_workers=8, shuffle=True, num_shards=device_num, shard_id=rank_id)

    def __getitem__(self, index):
        return self.dataset[0][index], self.dataset[1][index]

    def __len__(self):
        return len(self.dataset[0])


def create_dataset(dataset_dir, do_train, repeat_num=1, batch_size=32, target="Ascend"):
    ds = MsFaceDataset(dataset_dir)
    dataset = ds.dataset
    image_size = 112
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    if do_train:
        trans = [
            C.Decode(),
            C.RandomHorizontalFlip(prob=0.5),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    else:
        trans = [
            C.Decode(),
            C.Resize(256),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]
    type_cast_op = C2.TypeCast(mstype.int32)
    dataset = dataset.map(input_columns="image",
                          num_parallel_workers=8, operations=trans)
    dataset = dataset.map(input_columns="label", num_parallel_workers=8,
                          operations=type_cast_op)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(repeat_num)

    return dataset


def _get_rank_info():
    rank_size = int(os.environ.get("RANK_SIZE", 1))
    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0
    return rank_size, rank_id

if __name__ == "__main__":
    train_dataset = create_dataset(
        # dataset_dir="/home/data/xieguochen/dataset/AgeDataset/faces_webface_112x112_train", 
        dataset_dir="/home/data/xieguochen/dataset/AgeDataset/faces_emore_train",
        do_train=True, 
        repeat_num=1, 
        batch_size=128, 
        target="GPU")

    step = train_dataset.get_dataset_size()
    print(step)
