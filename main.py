import os
import numpy as np
from utils import ftp, download_ftp_tree
from config import config
from dcm_utils import load_dic, save_dic


def main():
    ftp.login(
        user=config['ftp']['user'],
        passwd=config['ftp']['password']
    )
    ftp.encoding = 'utf-8'
    root = os.getcwd()
    record_name = 'completed_patient_record.bin'

    completed_patients = {}
    try:
        completed_patients = load_dic(f'{root}/{record_name}')
    except Exception:
        pass

    print(completed_patients)

    path = config['ftp']['workpath']
    ftp.cwd(path)
    patient_list = ftp.nlst()

    for (patient_idx, patient) in enumerate(patient_list):

        ftp.cwd(path + patient)
        ct_list = ftp.nlst()

        if (patient in completed_patients) is False:
            completed_patients[patient] = []

        if os.path.exists(patient) is False:
            os.mkdir(patient)

        os.chdir(patient)

        for (ct_idx, ct) in enumerate(ct_list):

            if ct_idx > 0:
                break

            if (ct in completed_patients[patient]) is False:
                download_ftp_tree(ftp, ct, os.getcwd())
                completed_patients[patient].append(ct)
                save_dic(f'{root}/{record_name}', completed_patients)

                print(f'This CT image-{ct} in patient-{patient} proceeded completely.')

        os.chdir('..')

if __name__ == '__main__':
    main()
