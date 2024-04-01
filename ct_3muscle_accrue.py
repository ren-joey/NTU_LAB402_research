import numpy as np
from pathlib import Path
from PIL import Image

def ct_3muscle_accrue(in_path, out_path, id_range):
    in_path = Path(in_path)
    out_path = Path(out_path)

    for id in range(1, id_range + 1):
        r = np.array(Image.open(f'{in_path}/{id}_r.png'))
        g = np.array(Image.open(f'{in_path}/{id}_g.png'))
        b = np.array(Image.open(f'{in_path}/{id}_b.png'))
        new_img = np.array(r).tolist()
        for y, row in enumerate(new_img):
            for x, p in enumerate(row):
                new_img[y][x] = [
                    r[y][x],
                    g[y][x],
                    b[y][x]
                ]
        Image.fromarray(np.array(new_img)).save(
            f'{out_path}/{id}.png'
        )

if __name__ == '__main__':
    in_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/subtract'
    out_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/hstack'
    id_range = 911
    ct_3muscle_accrue(in_path, out_path, id_range)