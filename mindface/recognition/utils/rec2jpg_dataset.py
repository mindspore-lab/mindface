"""
rec format to jpg
"""
import os
import argparse
from skimage import io
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm


def main(dataset_path, output_dir):
    """
    main.
    """
    path_imgrec = os.path.join(dataset_path, 'train.rec')
    path_imgidx = os.path.join(dataset_path, 'train.idx')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])
    print('max_idx:', max_idx)
    for i in tqdm(range(max_idx)):
        header, s = recordio.unpack(imgrec.read_idx(i + 1))
        img = mx.image.imdecode(s).asnumpy()
        label = str(int(header.label))
        ids = str(i)

        label_dir = os.path.join(output_dir, label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)
        fname = f"Figure_{ids}.png"
        fpath = os.path.join(label_dir, fname)
        io.imsave(fpath, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='do dataset merge')
    # general
    parser.add_argument('--include', default='', type=str, help='')
    parser.add_argument('--output', default='', type=str, help='')
    args = parser.parse_args()
    main(args.include, args.output)
