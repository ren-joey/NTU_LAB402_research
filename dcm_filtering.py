import os
import re

import pydicom
from tqdm import tqdm

from plotter import img_plotter
from dcm_utils import dcm_tag_ck
from shutil import copy2


def dcm_filtering(dir):
    matched_tags = []
    matched_count = 0
    excluded_tags = []
    excluded_count = 0
    anomaly_count = 0
    proceed_dir = []
    anomaly_cp_path = os.path.join(os.getcwd(), 'anomalies')
    matched_cp_path = os.path.join(os.getcwd(), 'matched')
    os.makedirs(anomaly_cp_path, exist_ok=True)
    os.makedirs(matched_cp_path, exist_ok=True)


    for idx, (dir_path, dir_names, file_names) in enumerate(os.walk(dir)):

        # 確認路徑是 P{12}/AC{7}
        regex = '.*/RTMets_datasets/P\d{12}/AC\d{7}'
        res = re.fullmatch(regex, dir_path)
        tag_ck_pass = False

        # 如果路徑不對就找下一個 folder
        if res is None:
            continue
        # 如果該病患已經複製完畢，直接跳過
        else:
            path = os.path.normpath(dir_path + '/..')

            if path not in proceed_dir:
                regex = 'P\d{12}/AC\d{7}'
                matched = re.search(regex, dir_path).group()
                cp_target_path = os.path.join(matched_cp_path, matched)
            else:
                continue

        # 確認該路徑底下有沒有合格的 dicom 檔
        for file_idx, file in enumerate(
            tqdm(
                file_names,
                desc=f'checking dcm file tags'
            )
        ):
            regex = '^(?!\.).*\.dcm$'
            is_dcm = re.fullmatch(regex, file) is not None

            if is_dcm:
                file_path = os.path.join(dir_path, file)
                dcm = pydicom.dcmread(file_path, force=True)
                res, tagname, tag_pos = dcm_tag_ck(dcm)

                if res is True:
                    tag_ck_pass = True
                    break

        # 如果有找到合法檔案，將該路徑記錄起來
        if tag_ck_pass is True:
            proceed_dir.append(path)
            os.makedirs(cp_target_path, exist_ok=True)
            print(f'Current working dir: {cp_target_path}')
        # 如果沒有任何合法 tag 被找到，找下一個路徑
        else:
            continue


        for file_idx, file in enumerate(
            tqdm(
                file_names,
                desc=f'[{matched_count}, {excluded_count}, {anomaly_count}]'
            )
        ):

            regex = '^(?!\.).*\.dcm$'
            is_dcm = re.fullmatch(regex, file) is not None

            if is_dcm:
                file_path = os.path.join(dir_path, file)
                dcm = pydicom.dcmread(file_path, force=True)
                res, tagname, tag_pos = dcm_tag_ck(dcm)

                if res is True:
                    matched_count += 1
                    copy2(file_path, cp_target_path)

                    if tagname not in matched_tags:
                        matched_tags.append(tagname)

                elif res is False:
                    excluded_count += 1

                    if tagname not in excluded_tags:
                        excluded_tags.append(tagname)

                if tagname == '':
                    anomaly_count += 1
                    copy2(file_path, anomaly_cp_path)

    print(matched_tags)



if __name__ == '__main__':
    dir = '/Volumes/Transcend/RTMets_datasets'
    dcm_filtering(dir)
