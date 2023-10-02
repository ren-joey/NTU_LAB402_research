import os


def dcm_min_n_file(dir, min=30):
    dir = os.path.normpath(dir)
    anomaly_list = []

    for idx, (dir_path, dir_names, file_names) in enumerate(os.walk(dir)):
        regex = '.*/P\d{12}/AC\d{7}'
        res = re.fullmatch(regex, dir_path)

        if res is not None and len(file_names) < min:
            anomaly_list.append(dir_path)
