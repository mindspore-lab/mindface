import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

__all__=["create_dataset"]

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, target="Ascend", is_parallel=True):
    """
        create a train dataset

        Args:
            dataset_path(string): the path of dataset.
            do_train(bool): whether dataset is used for train or eval.
            repeat_num(int): the repeat times of dataset. Default: 1
            batch_size(int): the batch size of dataset. Default: 32
            target(str): the device target. Default: Ascend
            is_parallel(bool): training in parallel or not, Defualt: True

        Returns:
            dataset
    """
    if target == "Ascend":
        device_num, rank_id = _get_rank_info()
    else:
        if is_parallel:
            init("nccl")
            rank_id = get_rank()
            device_num = get_group_size()
        else:
            rank_id = 0
            device_num = 1

    if device_num == 1:
        ds = de.ImageFolderDataset(
            dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)

    image_size = 112
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    # define map operations
    if do_train:
        trans = [
            # C.RandomCropDecodeResize(image_size, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
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

    ds = ds.map(input_columns="image",
                num_parallel_workers=8, operations=trans)
    ds = ds.map(input_columns="label", num_parallel_workers=8,
                operations=type_cast_op)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)

    return ds


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
