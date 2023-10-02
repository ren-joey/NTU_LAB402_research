import os
import re
# import shutil
from os import path
# import pydicom
# import csv

from utils import delete_dir


def dcm2niix(dcm_dir, out_dir):
    dcm_dir = os.path.join(dcm_dir)
    anamoly_dir_list = []

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(dcm_dir)):

        regex = '[\\\/]P\d{12}[\\\/]AC\d{7}'
        res = re.search(regex, dir_path)

        if res is not None:
            if len(file_names) < 100:
                anamoly_dir_list.append(dir_path)

            # patient_path = os.path.normpath(dir_path + '/../..')
            niix_out_dir = os.path.normpath(out_dir + res.group() + '_niix')

            # delete_dir(niix_out_dir)
            os.makedirs(niix_out_dir, exist_ok=True)
            # os.system(f'dcm2niix -o {niix_out_dir} {dir_path}')
            command = f'C:\\Users\\axe09\\Desktop\\dcm2niix.exe -o {niix_out_dir} {dir_path}'
            os.system(command)

    print(anamoly_dir_list)

if __name__ == '__main__':
    # dcm_dir = '/Volumes/Transcend/matched'
    dcm_dir = 'D:\\matched'
    # out_dir = '/Users/joey_ren/Desktop/MS/Lab/Research/codes/niix'
    out_dir = 'C:\\Users\\axe09\\Desktop\\NTU_LAB402_research_code\\niix'
    dcm2niix(dcm_dir, out_dir)
