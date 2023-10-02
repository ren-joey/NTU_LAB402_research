import os
import pydicom as dicom
from os import path


def dcm_reader():
    local_dir = path.join(os.getcwd())
    # out_dir = local_dir + '/P214900000002/AC0000239-nixx'
    dic_dir = local_dir + '/P214900000002/AC0000239'
    meta_keys = [
        0x00020000,
        0x00020001,
        0x00020002,
        0x00020003,
        0x00020010,
        0x00020012,
        0x00020013
    ]

    for idx, (dir_path, dir_names, file_names) in enumerate(os.walk(dic_dir)):
        if idx > 1:
            break

        for file_idx, file in enumerate(file_names):
            # if file_idx > 1:
            #     break

            dcm_dir = path.join(dir_path, file)
            dcm = dicom.dcmread(dcm_dir, force=True)
            item = dcm[0x0008103E]
            # meta = dcm.file_meta.to_json()
            # item = meta[0x0008103E]

            if item is not None:
                print('===')
                print(item.value)
                print('===')

            # for key in meta_keys:
            #     try:
            #         print(key)
            #         # 0008, 103E
            #         item = meta.get_item(key)
            #         print(item)
            #     except Exception:
            #         pass


if __name__ == '__main__':
    dcm_reader()
