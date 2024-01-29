import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from argparse import ArgumentParser
import json
import math
from typing import Dict, Tuple
from collections import Counter
import scipy.ndimage
'''
取出VOI
'''

size = 224
z_size = 128

def GetNewPixelRange(row: Dict, shape: Tuple):
    # 本資料集的圖像大小都是 Slices × 512 × 512
    x_range, y_range, z_range = row['x_end'] - row['x_start'] + 1, row['y_end'] - row['y_start'] + 1, row['z_end'] - row['z_start'] + 1
    new_x, new_y, new_z = (row['x_start'], row['x_end']), (row['y_start'], row['y_end']), (row['z_start'], row['z_end'])
    if x_range < size:
        new_x = [row['x_start'] - math.ceil((size - x_range) / 2), row['x_end'] + math.floor((size - x_range) / 2)]
        if new_x[0] < 0:
            new_x[1] += -new_x[0]
            new_x[0] = 0
        elif new_x[1] > 511:
            new_x[0] -= new_x[1] - 511
            new_x[1] = 511
    if y_range < size:
        new_y = [row['y_start'] - math.ceil((size - y_range) / 2), row['y_end'] + math.floor((size - y_range) / 2)]
        if new_y[0] < 0:
            new_y[1] += -new_y[0]
            new_y[0] = 0
        elif new_y[1] > 511:
            new_y[0] -= new_y[1] - 511
            new_y[1] = 511
    if z_range < z_size:
        new_z = [row['z_start'] - math.ceil((z_size - z_range) / 2), row['z_end'] + math.floor((z_size - z_range) / 2)]
        if new_z[0] < 0:
            new_z[1] += -new_z[0]
            new_z[0] = 0
        elif new_z[1] > shape[0]:
            new_z[0] -= new_z[1] - shape[0]
            new_z[1] = shape[0]
    elif z_range > z_size:
        new_z = [row['z_start'] + math.ceil((z_range - z_size) / 2), row['z_end'] - math.floor((z_range - z_size) / 2)]

    return new_x, new_y, new_z

def main(args):
    # 讀 clinical data
    df = pd.read_excel(args.label_path, sheet_name=0, dtype=str)
    # change header
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    # 取需要的coulumns
    df = df[['Patient ID', 'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end']]
    df[['x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end']] = df[['x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end']].apply(pd.to_numeric)
    # dataframe to json
    df2: str = df.to_json(orient = 'records')
    df2: List = json.loads(df2)

    for row in df2:
        id = row['Patient ID']
        image = np.load(args.input_path / Path(id).with_suffix('.npy'))
        print(id, image.shape, image.dtype)

        x, y, z = GetNewPixelRange(row, image.shape)
        voi = image[z[0]:z[1]+1, y[0]:y[1]+1, x[0]:x[1]+1]

        print(id, voi.shape, voi.dtype)
        print("======================================================")
        np.save(args.output_path / Path(id).with_suffix('.npy'), voi)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--label_path', type=Path, default='../../Lung GSI patient list_20220818_查資料.xlsx')
    parser.add_argument('-i', '--input_path', type=Path, default='../../../data/C+_140/clip/')
    parser.add_argument('-o', '--output_path', type=Path, default='../../../data/C+_140/voi_128_224_224/')
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    main(args)