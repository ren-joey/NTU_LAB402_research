import numpy as np
from PIL import Image

def diode(img, target=255):
    img = np.array(img)

    for y, row in enumerate(img):
        for x, p in enumerate(row):
            if p > 0:
                img[y][x] = target
    return img

def ct_accrue(subtract_path, ct_path, out_path, id_range):
    for id in range(1, id_range + 1):
        ct = f'{ct_path}/{id}.png'
        trunk = f'{subtract_path}/{id}_accrue.png'
        mask = f'{subtract_path}/{id}_diode.png'
        ct = np.array(Image.open(ct))
        trunk = np.array(Image.open(trunk))
        mask = np.array(Image.open(mask))
        ct = diode(mask, 1) * ct
        new_img = ct.tolist()

        for y, row in enumerate(ct):
            for x, p in enumerate(row):
                new_img[y][x] = [
                    ct[y][x],
                    trunk[y][x],
                    mask[y][x]
                ]

        new_img = np.array(new_img)
        Image.fromarray(new_img).save(f'{out_path}/{id}.png')

if __name__ == '__main__':
    subtract_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/subtract'
    ct_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/xdata'
    out_path = '/Users/joey_ren/Desktop/MS/Lab402/research/code/datasets/muscle_group_segment/all/vstack'
    id_range = 911
    ct_accrue(subtract_path, ct_path, out_path, id_range)