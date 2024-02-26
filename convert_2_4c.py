import numpy as np
from pathlib import Path
import os
from os import listdir
from os.path import splitext, isfile, join
from PIL import Image
import torch

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def convert_2_4c(
    train_img_path,
    valid_img_path,
    train_mask_path,
    valid_mask_path,
    save_path
):
    train_img_path = Path(train_img_path)
    valid_img_path = Path(valid_img_path)
    train_mask_path = Path(train_mask_path)
    valid_mask_path = Path(valid_mask_path)

    train_img_ids = [splitext(file)[0] for file in listdir(train_img_path) if isfile(join(train_img_path, file)) and not file.startswith('.') and file.endswith(".png")]
    valid_img_ids = [splitext(file)[0] for file in listdir(valid_img_path) if isfile(join(valid_img_path, file)) and not file.startswith('.') and file.endswith(".png")]
    # train_mask_ids = [splitext(file)[0] for file in listdir(train_mask_path) if isfile(join(train_mask_path, file)) and not file.startswith('.') and file.endswith(".png")]
    # valid_mask_ids = [splitext(file)[0] for file in listdir(valid_mask_path) if isfile(join(valid_mask_path, file)) and not file.startswith('.') and file.endswith(".png")]

    # imgs = train_img_ids + valid_img_ids
    # size = len(imgs)
    # c4_imgs = np.empty(size + 1)
    for img_id in train_img_ids:
        mask_file = list(train_mask_path.glob(img_id + '.*'))
        img_file = list(train_img_path.glob(img_id + '.*'))

        assert len(mask_file) == 1
        assert len(img_file) == 1

        mask = np.array(load_image(mask_file[0]))
        img = np.array(load_image(img_file[0])).tolist()
        for y, rows in enumerate(img):
            for x, p in enumerate(rows):
                img[y][x] = [p]
        img = np.array(img)
        c4_img = np.concatenate((img, mask), axis=2)

        target = Path(save_path, f'{img_id}.npy')
        np.save(target, c4_img) # save

    for img_id in valid_img_ids:
        mask_file = list(valid_mask_path.glob(img_id + '.*'))
        img_file = list(valid_img_path.glob(img_id + '.*'))

        assert len(mask_file) == 1
        assert len(img_file) == 1

        mask = np.array(load_image(mask_file[0]))
        img = np.array(load_image(img_file[0])).tolist()
        for y, rows in enumerate(img):
            for x, p in enumerate(rows):
                img[y][x] = [p]
        img = np.array(img)
        c4_img = np.concatenate((img, mask), axis=2)

        target = Path(save_path, f'{img_id}.npy')
        np.save(target, c4_img) # save


if __name__ == '__main__':
    convert_2_4c(
        train_img_path='/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/train/xdata',
        valid_img_path='/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/validation/xdata',
        train_mask_path='/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/train/ydata',
        valid_mask_path='/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/validation/ydata',
        save_path='/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/4_channels'
    )
