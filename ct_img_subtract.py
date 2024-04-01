import torch
import os
import csv
from os.path import splitext, isfile, join
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd
from scipy import ndimage
import skimage.morphology, skimage.data


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)

def data_preparing(data_path):
    res = pd.read_csv(data_path)
    data = res.values
    header = res.columns.to_numpy()

    return data, header

def diode(img, target=255):
    img = np.array(img)

    for y, row in enumerate(img):
        for x, p in enumerate(row):
            if p > 0:
                img[y][x] = target
    return img

def erosion_dilation(img, times):
    diode_img = diode(img, target=1)
    for i in range(times):
        diode_img = ndimage.binary_erosion(diode_img)
    for i in range(times):
        diode_img = ndimage.binary_dilation(diode_img)

    # for i in range(times):
    #     diode_img = ndimage.binary_dilation(diode_img)
    # for i in range(times):
    #     diode_img = ndimage.binary_erosion(diode_img)
    diode_img = ndimage.binary_fill_holes(diode_img)
    diode_img = diode_img * 255

    # labels = skimage.morphology.label(diode_img)
    # print(np.unique(labels, return_counts=True))
    # raise
    # labelCount = np.bincount(labels.ravel())
    # background = np.argmax(labelCount)
    # img[labels != background] = 255

    # labels = (labels * 255).astype(np.uint8)
    # unique, _ = np.unique(labels, return_counts=True)
    # assert len(unique) == 2
    # assert labels.shape == img.shape

    return diode_img.astype(np.uint8)

def ct_img_subtract(ct_path, mask_path, out_path, spacing_map_path):
    ct_path = Path(ct_path)
    mask_path = Path(mask_path)
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    csv_path = Path(out_path, 'new_bg.csv')
    if csv_path.is_file():
        # confirm = input('remove current file (Y/N)?')
        os.remove(csv_path)
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['id', 'bg', 'bg_trunk'])
    row_data = []

    data, header = data_preparing(spacing_map_path)
    id_map = data[:, 1]
    spacing_map = data[:, 3]
    bg_map = data[:, 6]

    for idx, (dir_path, dir_names, file_names) in enumerate(os.walk(ct_path)):
        for filename in file_names:
            ct_file = Path(ct_path, filename)
            id = int(ct_file.stem)
            assert id_map[id - 1] == id
            bg, spacing = bg_map[id - 1], spacing_map[id - 1]

            mask_file = Path(mask_path, filename)
            ct = np.asarray(load_image(ct_file))
            ct_diode_255 = diode(ct)
            ct_fixed = erosion_dilation(ct_diode_255, 10)
            out_trunks = (ct_fixed.flatten() == 0).sum() * spacing * spacing
            print(out_trunks)
            mask = np.asarray(load_image(mask_file))
            r = diode(mask[:, :, 0], 1) * ct
            g = diode(mask[:, :, 1], 1) * ct
            b = diode(mask[:, :, 2], 1) * ct
            sub_r = Image.fromarray(r)
            sub_g = Image.fromarray(g)
            sub_b = Image.fromarray(b)
            sub_accrue = Image.fromarray(r + g + b)
            sub_r.save('{}/{}_r.png'.format(out_path, id))
            sub_g.save('{}/{}_g.png'.format(out_path, id))
            sub_b.save('{}/{}_b.png'.format(out_path, id))
            sub_accrue.save('{}/{}_accrue.png'.format(out_path, id))
            hole_filling_img = Image.fromarray(ct_fixed)
            hole_filling_img.save('{}/{}_diode.png'.format(out_path, id))
            row_data.append([
                id, bg, bg - out_trunks
            ])

    writer.writerows(row_data)
    f.close()

if __name__ == '__main__':
    ct_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/xdata'
    mask_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/ydata'
    out_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/subtract'
    spacing_map_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/niix_spacing.csv'
    ct_img_subtract(ct_path, mask_path, out_path, spacing_map_path)