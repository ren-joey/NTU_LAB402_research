import os
import re
import requests
from tqdm import tqdm


def niix2model(niix_dir, out_dir):
    url = 'http://localhost:5000/predict'
    niix_dir = os.path.join(niix_dir)
    anomoly_dir_list = []
    os.makedirs(out_dir, exist_ok=True)
    direction_name_list = ['seg', 'frontal']

    for dir_idx, (dir_path, dir_names, file_names) in enumerate(os.walk(niix_dir)):

        regex = 'P\d{12}[\\\/]AC\d{7}'
        res = re.search(regex, dir_path)

        if res is not None:
            patient = res.group()

            for file_idx, filename in enumerate(
                tqdm(
                    file_names,
                    desc=f'working on: {dir_path}'
                )
            ):
                regex = '^(?!\.).*\.nii$'
                is_nii = re.fullmatch(regex, filename) is not None

                if is_nii:
                    form = {
                        'image': (filename, open(os.path.join(dir_path, filename), 'rb'))
                    }
                    res = requests.post(url, files=form)

                    if res.status_code == 200:
                        id = res.json()['prediction']['id']
                        for direction_name in direction_name_list:
                            seg = requests.get(f'http://localhost:5000/uploads/{filename}_{direction_name}-{id}.jpg')
                            out_fullpath = os.path.join(out_dir, patient)
                            os.makedirs(out_fullpath, exist_ok=True)
                            img = open(f'{out_fullpath}/{filename}_{direction_name}-{id}.jpg', 'wb')
                            img.write(seg.content)
                            img.close()



if __name__  == '__main__':
    niix_dir = 'C:\\Users\\axe09\\Desktop\\NTU_LAB402_research_code\\niix'
    out_dir = 'C:\\Users\\axe09\\Desktop\\NTU_LAB402_research_code\\ydata'
    niix2model(niix_dir, out_dir)
