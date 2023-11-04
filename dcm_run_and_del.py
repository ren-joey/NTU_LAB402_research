import os
import re
import requests
# from tqdm import tqdm
import shutil
# import asyncio
# from aiohttp import ClientSession, MultipartWriter


# TODO:
# Must know how the python async system work


def dcm_run_and_del(
    in_dir,
    out_dir,
    dcm2niix_path,
    sarcopenia_server_url,
    pass_patients=[]
):
    in_dir = os.path.join(in_dir)
    out_dir = os.path.join(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # some function's constant
    direction_name_list = ['seg', 'frontal', 'slice', 'pred']
    patient_regex = '[\\\/]P\d{12}[\\\/]AC\d{7}'
    nii_ext_regex = '^(?!\.).*\.nii$'
    nii_out_dir = None

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(in_dir)):
        # FIXME:
        # Delete niix files
        # if nii_out_dir is not None:
        #     shutil.rmtree(nii_out_dir)

        # TODO: circuit breaker
        # if dir_idx > 2:
        #     break

        res = re.search(patient_regex, dir_path)

        # make sure that directory is a patient folder (named AC\d{7})
        if res is not None:
            patient = res.group()

            if patient in pass_patients:
                print(f'[PASS] patient:{patient}')
                continue

            nii_out_dir = os.path.normpath(out_dir + patient + '_niix')
            pred_out_dir = os.path.normpath(out_dir + patient)

            # convert and compress dcm files into niix file
            os.makedirs(nii_out_dir, exist_ok=True)
            os.makedirs(pred_out_dir, exist_ok=True)
            command = f'{dcm2niix_path} -o {nii_out_dir} {dir_path}'
            os.system(command)

            # list the proceeded niix files
            # and upload them to the sarcopenia-ai server individually
            for nii_file_name in os.listdir(nii_out_dir):

                # FIXME:
                # print(f'nii_file_name: {nii_file_name}')

                # identify niix files by file extension
                is_nii = re.fullmatch(nii_ext_regex, nii_file_name) is not None

                # if the file extension matched
                if is_nii:
                    nii_file_path = os.path.join(nii_out_dir, nii_file_name)

                    # FIXME:
                    # print(f'nii_file_path: {nii_file_path}')

                    form = {
                        'image': (nii_file_name, open(nii_file_path, 'rb'))
                    }

                    res = requests.post(sarcopenia_server_url, files=form)

                    if res.status_code == 200:
                        res = res.json()
                        if res == 1:
                            continue

                        id = res['prediction']['id']
                        z = res['prediction']['slice_z'] if 'slice_z' in res['prediction'] else 'inf'

                        for direction_name in direction_name_list:
                            for i in range(1, 3):
                                postfix = i if i != 1 else ''
                                seg = requests.get(f'http://localhost:5000/uploads/{nii_file_name}_{direction_name}-{id}{postfix}.jpg')

                                if seg.status_code == 200:
                                    img = open(f'{pred_out_dir}/{nii_file_name}_{direction_name}_slicez-{z}_{postfix}.jpg', 'wb')
                                    img.write(seg.content)
                                    img.close()


def detect_completed_patients(out_dir):
    patient_regex = '[\\\/]P\d{12}[\\\/]AC\d{7}'
    out_dir = os.path.join(out_dir)
    pass_patients = []

    for dir_path, dir_names, file_names in os.walk(out_dir):
        res = re.search(patient_regex, dir_path)

        if res is not None:
            pass_patients.append(res.group())

    return pass_patients


if __name__ == '__main__':
    dcm2niix_path = '"D:\\Users\\tsuyi\\Desktop\\dcm2niix.exe"'
    sarcopenia_server_url = 'http://localhost:5000/predict'
    out_dir = 'G:\\temp'

    pass_patients = detect_completed_patients(out_dir)

    dcm_run_and_del(
        'G:\\RTMets_datasets',
        out_dir,
        dcm2niix_path,
        sarcopenia_server_url,
        pass_patients
    )