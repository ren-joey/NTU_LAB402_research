import os
import glob
import csv
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pydicom
from pathlib import Path
from typing import Dict, List
from pathlib import Path
import glob
'''
Dicom轉npy
'''
AI_path = '../../Lung GSI patient list_20220818_查資料.xlsx'

def load_scan(path):
    files_name = [s for s in os.listdir(path)]
    slice_dicom = []
    for file in files_name:
        if not (file.endswith('bmp') or file.endswith('xlsx')):
            slice_dicom.append(file)
    slice_dicom.sort()
    slices = [pydicom.dcmread(path / Path(s), force=True) for s in slice_dicom]	

    return slices

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])  #[D,512,512], dtype=uint16
    # 轉為int16，int16是ok的，因為所有數值應 <32k
    image = image.astype(np.int16)
    # 轉換為CT值 (HU單位)
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        if slope != 1:  #HU = slope * slice + intercept
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)

        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)	

# 獲取要使用的病人號碼
_excel = pd.read_excel(AI_path, sheet_name=0, dtype=str)
# change header
_excel.columns = _excel.iloc[0]
_excel = _excel.drop(_excel.index[0])
case_list = _excel['Patient ID'].values.tolist()

# 獲取所有病歷的CT位置
filepaths: Dict = {filepath.name.split('_')[0]:filepath for filepath in Path('../../../Dicom').iterdir()}

hash_table = {
    'C+_Mono_40*': Path('C+_40'),
    'C+_Mono_50*': Path('C+_50'),
    'C+_Mono_60*': Path('C+_60'),
    'C+_Mono_70*': Path('C+_70'),
    'C+_Mono_80*': Path('C+_80'),
    'C+_Mono_90*': Path('C+_90'),
    'C+_Mono_100*': Path('C+_100'),
    'C+_Mono_110*': Path('C+_110'),
    'C+_Mono_120*': Path('C+_120'),
    'C+_Mono_130*': Path('C+_130'),
    'C+_Mono_140*': Path('C+_140'),

    'C-_Mono_40*': Path('C-_40'),
    'C-_Mono_50*': Path('C-_50'),
    'C-_Mono_60*': Path('C-_60'),
    'C-_Mono_70*': Path('C-_70'),
    'C-_Mono_80*': Path('C-_80'),
    'C-_Mono_90*': Path('C-_90'),
    'C-_Mono_100*': Path('C-_100'),
    'C-_Mono_110*': Path('C-_110'),
    'C-_Mono_120*': Path('C-_120'),
    'C-_Mono_130*': Path('C-_130'),
    'C-_Mono_140*': Path('C-_140'),

    'C-Mono_40*': Path('C-_40'),
    'C-Mono_50*': Path('C-_50'),
    'C-Mono_60*': Path('C-_60'),
    'C-Mono_70*': Path('C-_70'),
    'C-Mono_80*': Path('C-_80'),
    'C-Mono_90*': Path('C-_90'),
    'C-Mono_100*': Path('C-_100'),
    'C-Mono_110*': Path('C-_110'),
    'C-Mono_120*': Path('C-_120'),
    'C-Mono_130*': Path('C-_130'),
    'C-Mono_140*': Path('C-_140'),

    'C_Mono_40*': Path('C-_40'),
    'C_Mono_50*': Path('C-_50'),
    'C_Mono_60*': Path('C-_60'),
    'C_Mono_70*': Path('C-_70'),
    'C_Mono_80*': Path('C-_80'),
    'C_Mono_90*': Path('C-_90'),
    'C_Mono_100*': Path('C-_100'),
    'C_Mono_110*': Path('C-_110'),
    'C_Mono_120*': Path('C-_120'),
    'C_Mono_130*': Path('C-_130'),
    'C_Mono_140*': Path('C-_140'),
}

# 對於要使用的病人，讀取dicom
count = 0 
for ct_case in case_list:
    if ct_case in filepaths:
        print(ct_case)
        ct_dir = filepaths[ct_case]
        for CT_dir in hash_table.keys():
            dirs = Path(ct_dir).glob(CT_dir)
            ct_path = ''
            for dir in dirs:
                ct_path = dir
            try:
                patient = load_scan(ct_path)
                patient_pixels = get_pixels_hu(patient)
                SAVE_FOLDER = Path('../../data') / hash_table[CT_dir] / Path('raw_data')
                SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
                np.save(os.path.join(SAVE_FOLDER, str(ct_case)+'.npy'), patient_pixels)
            except:
                print("Not in ", CT_dir)
        count += 1

print(count)