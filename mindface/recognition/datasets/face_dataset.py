"""
face_dataset
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset.engine as de
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.transforms.c_transforms as C2
from mindspore.communication.management import init, get_rank, get_group_size

__all__=["create_dataset"]

def create_dataset(dataset_path, do_train, repeat_num=1, batch_size=32, augmentation=None,
                    target="Ascend", is_parallel=True):
    """
    Create a train dataset.

    Args:
        dataset_path (String): The path of dataset.
        do_train (Bool): Whether dataset is used for train or eval.
        repeat_num (Int): The repeat times of dataset. Default: 1.
        batch_size (Int): The batch size of dataset. Default: 32.
        augmentation (List): Data augmentation. Default: None.
        target (String): The device target. Default: "Ascend".
        is_parallel (Bool): Parallel training parameters. Default: True.

    Returns:
        ds (Object), data loader.

    Examples:
        >>> training_dataset = "/path/to/face_dataset"
        >>> train_dataset = create_dataset(dataset_path=training_dataset, do_train=True)
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
        data_set = de.ImageFolderDataset(
            dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        data_set = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)

    image_size = 112
    mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
    std = [0.5 * 255, 0.5 * 255, 0.5 * 255]

    # define map operations
    if do_train:
        if augmentation:
            trans = augmentation
        else:
            trans = [
                C.Decode(),
                C.RandomHorizontalFlip(prob=0.5),
                C.Normalize(mean=mean, std=std),
                C.HWC2CHW()
            ]
    else:
        trans = [
            C.Decode(),
            C.Resize(112),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image",
                num_parallel_workers=8, operations=trans)
    data_set = data_set.map(input_columns="label", num_parallel_workers=8,
                operations=type_cast_op)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


def _get_rank_info():
    """
    Get rank size and rank id.
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id
