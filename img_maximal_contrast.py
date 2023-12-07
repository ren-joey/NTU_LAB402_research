import numpy as np
from PIL import Image
from os.path import splitext, isfile, join
from os import listdir
from pathlib import Path

def img_anomaly_correct(file):
    file = Path(file)
    mask = np.asarray(Image.open(file))
    mask = mask.copy()
    modified = False
    for y, row in enumerate(mask):
        for x, pixel in enumerate(row):
            _ = pixel.tolist()
            if _ == [255, 0, 255]:
                modified = True
                mask[y][x] = np.array([255, 0, 0])
            elif _ == [255, 255, 0]:
                modified = True
                mask[y][x] = np.array([255, 0, 0])
            elif _ == [0, 255, 255]:
                modified = True
                mask[y][x] = np.array([0, 0, 255])
            elif _ == [255, 255, 255]:
                modified = True
                mask[y][x] = np.array([255, 0, 0])

    if modified == True:
        Image.fromarray(mask).save(Path(file.parent, f'{file.stem}.png'))


def img_maximal_contrast(file):
    file = Path(file)
    mask = np.asarray(Image.open(file))
    mask = mask.copy()
    for y, row in enumerate(mask):
        for x, pixel in enumerate(row):
            for z, value in enumerate(pixel):
                if value == 0 or value == 255:
                    continue
                elif value > 127:
                    mask[y][x][z] = 255
                elif value <= 127:
                    mask[y][x][z] = 0

    Image.fromarray(mask).save(Path(file.parent, f'{file.stem}.png'))



if __name__ == '__main__':
    path = "/Users/joey_ren/Desktop/muscle_group_segment/ydata"
    files = [file for file in listdir(path) if isfile(join(path, file)) and not file.startswith('.') and file.endswith(".png")]
    for idx, file in enumerate(files):
        file_path = join(path, file)
        img_maximal_contrast(file_path)
        img_anomaly_correct(file_path)