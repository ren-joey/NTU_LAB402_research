import os
import pickle
import re

import pydicom

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

if __name__ == '__main__':
    # dcm_dir = '/Volumes/Transcend/RTMets_datasets/P214900000003/AC0003002/FO-5677511027863284789.dcm'
    # dcm = pydicom.dcmread(dcm_dir, force=True)
    # res, tagname, _ = dcm_tag_ck(dcm)
    # img_plotter(dcm)
    res, tagname, _ = dcm_tag_ck(tagname='Localizers')
    print(res)
    print(tagname)