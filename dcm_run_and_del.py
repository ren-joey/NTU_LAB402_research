import os
import re
import requests
from tqdm import tqdm

dcm2niix_path = 'C:\\Users\\axe09\\Desktop\\dcm2niix.exe'
sarcopenia_server_url = 'http://localhost:5000/predict'

def dcm_run_and_del(
    in_dir,
    out_dir,
    dcm2niix_path,
    sarcopenia_server_url
):
    in_dir = os.path.join(in_dir)
    out_dir = os.path.join(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # some function's constant
    direction_name_list = ['seg', 'frontal']
    patient_regex = '[\\\/]P\d{12}[\\\/]AC\d{7}'
    nii_ext_regex = '^(?!\.).*\.nii$'

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(in_dir)):

        # TODO: circuit breaker
        if dir_idx > 5:
            break

        res = re.search(patient_regex, dir_path)

        # make sure that directory is a patient folder (named AC\d{7})
        if res is not None:
            patient = res.group()
            niix_out_dir = os.path.normpath(out_dir + res.group() + '_niix')

            # convert and compress dcm files into niix file
            os.makedirs(niix_out_dir, exist_ok=True)
            command = f'{dcm2niix_path} -o {niix_out_dir} {dir_path}'
            os.system(command)

            # list the proceeded niix files
            # and upload them to the sarcopenia-ai server individually
            for nii_file_names in os.list(niix_out_dir):

                # identify niix files by file extension
                is_nii = re.fullmatch(nii_ext_regex, nii_file_names) is not None

                ## TODO:
                ## sarcopenia-ai 那邊還要改成紫色飽滿

                # if the file extension matched
                if is_nii:
                    form = {
                        'image': (nii_file_names, open(os.path.join(dir_path, nii_file_names), 'rb'))
                    }
                    res = requests.post(sarcopenia_server_url, files=form)

                    if res.status_code == 200:
                        res = res.json()
                        id = res['prediction']['id']
                        z = res['prediction']['slice_z'] if 'slice_z' in res['prediction'] else 'inf'

                        for direction_name in direction_name_list:
                            for i in range(1, 3):
                                postfix = i if i != 1 else ''
                                seg = requests.get(f'http://localhost:5000/uploads/{nii_file_names}_{direction_name}-{id}{postfix}.jpg')

                                if seg.status_code == 200:
                                    out_fullpath = os.path.join(out_dir, patient)
                                    os.makedirs(out_fullpath, exist_ok=True)
                                    img = open(f'{out_fullpath}/{nii_file_names}_{direction_name}_slicez-{z}.jpg', 'wb')
                                    img.write(seg.content)
                                    img.close()


if __name__ == '__main__':
    dcm_run_and_del(
        'E:\\RTMets_datasets\\P214900000002',
        'E:\\temp',
        dcm2niix_path, sarcopenia_server_url
    )