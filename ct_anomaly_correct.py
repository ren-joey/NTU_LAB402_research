import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def data_preparing(data_path):
    res = pd.read_csv(data_path)
    data = res.values
    header = res.columns.to_numpy()

    return data, header

def ct_anomaly_correct(in_path, spacing_map_path):
    in_path = Path(in_path)
    data, header = data_preparing(spacing_map_path)
    id_map = data[:, 1]
    spacing_map = data[:, 3]
    bg_map = data[:, 6]
    # pd.read_csv()

    csv_path = Path(in_path, 'new_bg.csv')
    if csv_path.is_file():
        # confirm = input('remove current file (Y/N)?')
        os.remove(csv_path)
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['id', 'bg', 'bg_trunk'])
    row_data = []

    for dir_path, dir_names, file_names in os.walk(in_path):
        for filename in file_names:
            filepath = Path(in_path, filename)

            if filepath.suffix != '.png':
                continue

            id = int(filepath.stem.split('_')[0])
            assert id == id_map[id - 1]
            spacing = spacing_map[id - 1]
            bg = bg_map[id - 1]

            ct = np.array(Image.open(filepath))

            for y, row in enumerate(ct):
                for x, p in enumerate(row):
                    if p > 127:
                        ct[y][x] = 255
                    else:
                        ct[y][x] = 0
            Image.fromarray(ct).save(filepath)

            bg_trunk = (ct.flatten() == 0).sum() * spacing * spacing
            row_data.append([
                id,
                bg,
                bg - bg_trunk
            ])

    writer.writerows(row_data)
    f.close()

if __name__ == '__main__':
    in_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/anomaly'
    spacing_map_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/niix_spacing.csv'
    ct_anomaly_correct(in_path, spacing_map_path)