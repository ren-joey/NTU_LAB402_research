import pandas as pd
import os
import numpy as np
np.set_printoptions(suppress=True)
files = list(os.listdir(r"D:\Users\tsuyi\Desktop\NTU_LAB402_research_code\utils\ROC_Curve\data"))

for file in files:
    print(file)
    if file == '.DS_Store':
        continue
    mali_indexes = []
    benign_indexes = []

    data = pd.read_csv(os.path.join(r"D:\Users\tsuyi\Desktop\NTU_LAB402_research_code\utils\ROC_Curve\data",file))

    for ind, value in enumerate(data['label']):
        if value == 1:
            mali_indexes.append(ind)
        else:
            benign_indexes.append(ind)


    ben_val = list(data['prob'][benign_indexes])
    mal_val = list(data['prob'][mali_indexes])

    path = file + '.txt'
    f = open(os.path.join(r"D:\Users\tsuyi\Desktop\NTU_LAB402_research_code\utils\ROC_Curve\txt", path), 'w')
    f.write('method 1'+'\n')
    f.write('Large'+'\n'+'\n')
    for ben in ben_val:
        # print(str(np.array([ben]))[1:-1])
        f.write(str(np.array([ben]))[1:-1]+'\n')
    f.write('*\n')
    for mali in mal_val:
        f.write(str(np.array([mali]))[1:-1]+'\n')
    f.write('*\n')
    f.close()
    # break

