import os
import re
from pathlib import Path
import SimpleITK as sitk
import pydicom as dicom
import csv
from PIL import Image
import numpy as np

row = ['id', 'patient_id', 'scan_id', 'spacing_0', 'spacing_1', 'spacing_2', 'bg', 'erector_spinae', 'psoas', 'abdominis', 'l3_muscle']

def img_anomaly_correct(img):
    mask = np.asarray(img)
    mask = mask.copy()
    for y, row in enumerate(mask):
        for x, pixel in enumerate(row):
            _ = pixel.tolist()
            if _ == [255, 0, 255]:
                mask[y][x] = np.array([255, 0, 0])
            elif _ == [255, 255, 0]:
                mask[y][x] = np.array([255, 0, 0])
            elif _ == [0, 255, 255]:
                mask[y][x] = np.array([0, 0, 255])
            elif _ == [255, 255, 255]:
                mask[y][x] = np.array([255, 0, 0])

    return mask


def img_maximal_contrast(img):
    mask = np.asarray(img)
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

    return img_anomaly_correct(mask)


def mapYDataToNiixFiles(ydata, niix, out):
    ydata_dir = Path(ydata)
    niix_dir = Path(niix)
    out_dir = Path(out)
    count = 0

    csv_path = Path(out, 'niix_spacing.csv')
    if csv_path.is_file():
        # confirm = input('remove current file (Y/N)?')
        os.remove(csv_path)
    f = open(csv_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(row)
    data = []

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(ydata_dir)):
        regex = 'P\d{12}$'
        res = re.search(regex, dir_path)

        if res is not None:
            patient = res.group()
            regex = '^(?!\.).*\.png$'
            r = re.compile(regex)
            files = list(filter(r.match, file_names))

            if len(files) != 0:
                filename = files[0]
                print(Path(dir_path, filename))
                ct = Image.open(Path(dir_path, filename))
                ct = img_maximal_contrast(ct)
                pixels = ct.shape[0] * ct.shape[1]
                colors, counts = np.unique(ct.reshape(-1, 3), axis=0, return_counts=1)
                bg = 0
                psoas = 0
                erector_spinae = 0
                abdominis = 0
                for idx, color in enumerate(colors):
                    if (color == [0, 0, 0]).all():
                        bg = counts[idx]
                    elif (color == [255, 0, 0]).all():
                        erector_spinae = counts[idx]
                    elif (color == [0, 255, 0]).all():
                        psoas = counts[idx]
                    elif (color == [0, 0, 255]).all():
                        abdominis = counts[idx]

                niix_name = filename.split('_pred_slicez')[0]
                regex = 'AC\d{7}'
                res = re.search(regex, filename)
                ct_id = res.group()

                niix_matched_dir = Path(niix_dir, patient, f'{ct_id}_niix', niix_name)

                images = sitk.ReadImage(niix_matched_dir)
                spacing = images.GetSpacing()

                if niix_matched_dir.is_file():
                    count += 1

                rowdata = [
                    count,
                    patient,
                    ct_id,
                    spacing[0],
                    spacing[1],
                    spacing[2],
                    bg * spacing[0] * spacing[0],
                    erector_spinae * spacing[0] * spacing[0],
                    psoas * spacing[0] * spacing[0],
                    abdominis * spacing[0] * spacing[0],
                    (pixels - bg) * spacing[0] * spacing[0]
                ]
                print(rowdata)
                data.append(rowdata)
    writer.writerows(data)
    f.close()
    print(count)

if __name__ == '__main__':
    ydata = 'G:/L3'
    niix = 'G:/temp'
    out = 'D:/Users/tsuyi/Desktop/temp'
    mapYDataToNiixFiles(ydata, niix, out)