import os
import re
import requests
from tqdm import tqdm


def dcm_run_and_del(in_dir, out_dir):
    in_dir = os.path.join(in_dir)
    out_dir = os.path.join(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(in_dir)):
        if dir_idx > 5:
            break

        regex = '[\\\/]P\d{12}[\\\/]AC\d{7}'
        res = re.search(regex, dir_path)

        if res is not None:
            niix_out_dir = os.path.normpath(out_dir + res.group() + '_niix')

            # delete_dir(niix_out_dir)
            os.makedirs(niix_out_dir, exist_ok=True)
            # os.system(f'dcm2niix -o {niix_out_dir} {dir_path}')
            command = f'C:\\Users\\axe09\\Desktop\\dcm2niix.exe -o {niix_out_dir} {dir_path}'
            os.system(command)

            for nii_file_names in os.list(niix_out_dir):
                regex = '^(?!\.).*\.nii$'
                is_nii = re.fullmatch(regex, nii_file_names) is not None

                ## TODO::
                ## 以下還沒改完
                ## sarcopenia-ai 那邊還要改成紫色飽滿
                if is_nii:
                    form = {
                        'image': (nii_file_names, open(os.path.join(dir_path, nii_file_names), 'rb'))
                    }
                    res = requests.post(url, files=form)

                    if res.status_code == 200:
                        res = res.json()
                        id = res['prediction']['id']
                        z = res['prediction']['slice_z'] if 'slice_z' in res['prediction'] else 'inf'

                        for direction_name in direction_name_list:
                            seg = requests.get(f'http://localhost:5000/uploads/{nii_file_names}_{direction_name}-{id}.jpg')

                            if seg.status_code == 200:
                                out_fullpath = os.path.join(out_dir, patient)
                                os.makedirs(out_fullpath, exist_ok=True)
                                img = open(f'{out_fullpath}/{nii_file_names}_{direction_name}_slicez-{z}.jpg', 'wb')
                                img.write(seg.content)
                                img.close()


if __name__ == '__main__':
    dcm_run_and_del('E:\\RTMets_datasets\\P214900000002', 'E:\\temp')