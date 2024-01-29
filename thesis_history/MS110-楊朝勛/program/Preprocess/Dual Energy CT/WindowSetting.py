import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List
from argparse import ArgumentParser
'''
對raw_VOI clip
'''
MIN_BOUND = -1000
MAX_BOUND = 400

def main(args):
    files = glob.glob(os.path.join(args.input_path, '*.npy'))
    print('共有:', len(files), '個檔案')
    for file in files:
        id = Path(file).stem
        image = np.load(file)
        print(id, image.min(), image.max())

        image[image > 400] = 400
        image[image < -1000] = -1000

        np.save(args.output_path / Path(id).with_suffix('.npy'), image)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=Path, default='../../../data/C+_140/raw_data/')
    parser.add_argument('-o', '--output_path', type=Path, default='../../../data/C+_140/clip/')
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)
    main(args)