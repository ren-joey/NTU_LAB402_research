import os
import pickle
import re
import shutil
import pydicom
import cv2
from plotter import img_plotter


def save_dic(name, dic):
    root = os.getcwd()
    save_dir = os.path.join(root, name + '.bin')
    postfix = 0
    while os.path.exists(save_dir):
        postfix += 1
        save_dir = os.path.join(root, f'{name} ({postfix}).bin')

    # save dictionary to person_data.pkl file
    with open(save_dir, 'wb') as fp:
        pickle.dump(dic, fp)


def load_dic(name):
    with open(f'{name}.bin', 'rb') as fp:
        file = pickle.load(fp)
    return file


def dcm_tag_ck(dcm=None, tagname=None):
    if dcm is None:
        tag_pos_list = [None]
    else:
        tag_pos_list = [0x0008103E, 0x00400254]

    for tag_pos in tag_pos_list:
        if dcm is not None:
            try:
                tagname = dcm[tag_pos].value
            except Exception:
                continue

        neg_reg = '(SAG|BRAIN|COR|Localizers|Processed Images|Head|Reformatted|Axial|DELAY|Synthetic series containing IMPAX MVF Markup)'
        pos_reg = '(?!S)A(?!G)'
        neg_test = re.search(neg_reg, tagname, flags=re.IGNORECASE) is None
        pos_test = re.search(pos_reg, tagname, flags=re.IGNORECASE) is not None

        if neg_test and pos_test:
            return neg_test and pos_test, tagname, tag_pos

    return False, tagname, tag_pos_list[len(tag_pos_list) - 1]

def cptree(src, dst):

    def keep_only_dirs(path, files):
        to_ignore = [
            fname for fname in files
            if not os.path.isdir(os.path.join(path, fname))
            ]
        return to_ignore

    # This works for python3 (<3.8), BUT the target directory MUST not exist
    # shutil.copytree(src, dst, ignore=keep_only_dirs)

    # For python 3.8 you can use following code (Here the target directory can
    # already exist and only new directories will be added
    shutil.copytree(src, dst, ignore=keep_only_dirs, dirs_exist_ok=True)

def dcm2png(dcm_dir):
    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(dcm_dir)):

        regex = '^(?!\.).*\.dcm$'

        for file_idx, file in enumerate(file_names):
            is_dcm = re.fullmatch(regex, file) is not None

            if is_dcm:
                file_path = os.path.join(dir_path, file)
                ds = pydicom.read_file(file_path)
                img = ds.pixel_array # get image array
                cv2.imwrite(os.path.join(dir_path, file.replace('.dcm', '.png')), img)

if __name__ == '__main__':
    ### dcm file plot
    # dcm_dir = '/Volumes/Transcend/RTMets_datasets/P214900000003/AC0003002/FO-5677511027863284789.dcm'
    # dcm = pydicom.dcmread(dcm_dir, force=True)
    # res, tagname, _ = dcm_tag_ck(dcm)
    # img_plotter(dcm)

    ### tag check test
    # res, tagname, _ = dcm_tag_ck(tagname='Localizers')
    # print(res)
    # print(tagname)

    ### cptree test
    # cptree('E:\\RTMets_datasets', 'C:\\Users\\axe09\\Desktop\\L3')

    ### dcm2png test
    dcm2png('C:\\Users\\axe09\\Desktop\\L3')