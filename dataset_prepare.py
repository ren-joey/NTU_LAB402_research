import os
from pathlib import Path
import shutil
from PIL import Image


def dataset_prepare(work_dir, out_dir):
    work_dir = Path(work_dir)
    out_dir = Path(out_dir)

    ct_xdata = out_dir.joinpath('muscle_group_segment\\xdata')
    ct_ydata = out_dir.joinpath('muscle_group_segment\\ydata')
    grouping_xdata = out_dir.joinpath('grouping\\xdata')
    grouping_ydata = out_dir.joinpath('grouping\\ydata')

    os.makedirs(ct_xdata, exist_ok=True)
    os.makedirs(ct_ydata, exist_ok=True)
    os.makedirs(grouping_xdata, exist_ok=True)
    os.makedirs(grouping_ydata, exist_ok=True)

    idx = 1

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(work_dir)):
        for file in file_names:
            y_file_path = Path(dir_path, file)

            if y_file_path.suffix == '.png':
                name = y_file_path.name.split('.png')[0]
                names = name.split('_pred_')

                shutil.copyfile(
                    y_file_path,
                    Path(ct_ydata, f'{idx}.png')
                )
                shutil.copyfile(
                    y_file_path,
                    Path(grouping_ydata, f'{idx}.png')
                )

                grouping_xdata_img = Image.open(Path(dir_path, f'{name}.jpg'))
                grouping_xdata_img.save(Path(grouping_xdata, f'{idx}.png'))

                ct_xdata_img = Image.open(Path(dir_path, f'{names[0]}_slice_{names[1]}.jpg'))
                ct_xdata_img.save(Path(ct_xdata, f'{idx}.png'))

                idx += 1

                # nii_name = name.split('_pred_')[0]
                # print(nii_name)


if __name__ == '__main__':
    work_dir = 'D:\\Users\\tsuyi\\Desktop\\L3'
    out_dir = 'D:\\Users\\tsuyi\\Desktop\\datasets'

    dataset_prepare(work_dir, out_dir)