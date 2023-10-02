import os
import re
import shutil

import pydicom
from tqdm import tqdm
from dcm_utils import dcm_tag_ck, save_dic, load_dic
from utils import delete_dir_recursion, delete_dir


def dcm_rm_dir_searcher(tar_dir, max_cts=6, record_name='remove_list'):
    tar_dir = os.path.join(tar_dir)
    valid_folder_count = 0
    current_working_dir = ''
    remove_dir_list = []
    anomaly_dir_list = []
    root = os.getcwd()

    try:
        dic = load_dic(os.path.join(root, record_name))

        condition = True

        while condition:
            print(dic)
            user_input = input('Record file is existing, would you like to overwrite it? (yes/no): ')

            yes_choices = ['yes', 'y']
            no_choices = ['no', 'n']

            if user_input.lower() in yes_choices:
                condition = False
                remove_dir_list = dic['remove_dir_list']
                anomaly_dir_list = dic['anomaly_dir_list']
            elif user_input.lower() in no_choices:
                return
    except FileNotFoundError:
        pass

    for idx, (dir_path, dir_names, file_names) in enumerate(os.walk(tar_dir)):

        regex = '.*/RTMets_datasets/P\d{12}'
        res = re.fullmatch(regex, dir_path)

        if res is not None:
            valid_folder_count = 0
            current_working_dir = dir_path
            continue

        regex = '.*/RTMets_datasets/P\d{12}/AC\d{7}'
        res = re.fullmatch(regex, dir_path)

        if res is None:
            anomaly_dir_list.append(dir_path)
            continue

        elif valid_folder_count >= max_cts:
            remove_dir_list.append(dir_path)
            continue

        for file_idx, file in enumerate(
            tqdm(
                file_names,
                desc=f'working dir: {current_working_dir}'
            )
        ):
            regex = '^(?!\.).*\.dcm$'
            is_dcm = re.fullmatch(regex, file) is not None

            if is_dcm:
                file_path = os.path.join(dir_path, file)
                dcm = pydicom.dcmread(file_path, force=True)
                res, tagname, tag_pos = dcm_tag_ck(dcm)

                if res is True:
                    valid_folder_count += 1
                    print(dir_path, valid_folder_count, tag_pos, tagname)
                    break
                # else:
                #     print(f'invalid tag: {tagname}')
            elif file_idx + 1 == len(file_names):
                print('no any matched dicom file:')
                print(dir_path)
                # remove_dir_list.append(dir_path)

    dic = {
        'remove_dir_list': remove_dir_list,
        'anomaly_dir_list': anomaly_dir_list
    }
    print(dic)
    save_dic(os.path.join(root, record_name), dic)


def dcm_remover(record_name='remove_list', max_cts=0):
    root = os.getcwd()

    dic = load_dic(os.path.join(root, record_name))
    remove_dir_list = dic['remove_dir_list']

    condition = True

    while condition:
        user_input = input(f'This procedure will delete {len(remove_dir_list)} folders, are you sure? (yes/no): ')

        yes_choices = ['yes', 'y']
        no_choices = ['no', 'n']

        if user_input.lower() in yes_choices:
            condition = False
        elif user_input.lower() in no_choices:
            return

    for idx, dir_path in enumerate(remove_dir_list):
        regex = '.*/RTMets_datasets/P\d{12}/AC\d{7}'
        res = re.fullmatch(regex, dir_path)

        if res is None:
            continue
        else:
            patient_dir = os.path.normpath(dir_path + '/..')
            folder_n = len(os.listdir(patient_dir))

            if folder_n <= max_cts:
                print(patient_dir, folder_n)
                continue

            # cts = os.listdir(os.path.normpath(dir_path + '/..'))
            # print(len(cts))

            delete_dir(dir_path)


if __name__ == '__main__':
    filename = 'remove_list'
    dir = '/Volumes/Transcend/RTMets_datasets'
    dcm_rm_dir_searcher(dir, max_cts=6, record_name='remove_list (3)')
    dcm_remover(max_cts=6, record_name='remove_list (3)')


