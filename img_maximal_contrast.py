import numpy as np
from PIL import Image
from os.path import splitext, isfile, join
from os import listdir
from pathlib import Path



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
    print(file)
    Image.fromarray(mask).save(Path(file.parent, f'{file.stem}.png'))

    # print(mask)
    # print(mask.ndim)
    # mask = mask.reshape(-1, mask.shape[-1])
    # print(np.unique(mask, axis=0))


if __name__ == '__main__':
    path = "G:\\datasets\\muscle_group_segment\\ydata"
    files = [file for file in listdir(path) if isfile(join(path, file)) and not file.startswith('.') and file.endswith(".png")]
    for idx, file in enumerate(files):
        unique_mask_values(join(path, file))