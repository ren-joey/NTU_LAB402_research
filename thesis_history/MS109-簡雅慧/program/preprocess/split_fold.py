import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import os


label_path = 'D:/3D_ABUS/all_label/'

save_path = 'D:/3D_ABUS/folds_in_128' # folds_res
os.makedirs(save_path, exist_ok=True)

df_b = pd.read_excel('D:/3D_ABUS/case.xlsx', sheet_name='Ben')
df_m = pd.read_excel('D:/3D_ABUS/case.xlsx', sheet_name='Mal')
id_b = np.array(df_b['ID'])
BI_b = np.array(df_b['BI-RADS'])
id_m = np.array(df_m['ID'])
BI_m = np.array(df_m['BI-RADS'])
bm_b = []
bm_m = []
bi_b = []
bi_m = []
for sub in range(len(id_b)):
    bm_b.append(0)
    BI = BI_b[sub]
    if BI == 3:
        bi_b.append(3)
    elif BI == 5:
        bi_b.append(5)
    else:
        bi_b.append(4)
for sub in range(len(id_m)):
    bm_m.append(1)
    BI = BI_m[sub]
    if BI == 3:
        bi_m.append(3)
    elif BI == 5:
        bi_m.append(5)
    else:
        bi_m.append(4)
bm_b = np.array(bm_b)
bm_m = np.array(bm_m)
bi_b = np.array(bi_b)
bi_m = np.array(bi_m)

ben_label = np.load(label_path + 'ben_lex_use.npy')
mal_label = np.load(label_path + 'mal_lex_use.npy')

kfold = KFold(n_splits=5, shuffle=True)
ben_idx = []
mal_idx = []

for train_index, test_index in kfold.split(ben_label):
    ben_idx.append(test_index)

for train_index, test_index in kfold.split(mal_label):
    mal_idx.append(test_index)

for i in range(5):
    print(i)

    ID = np.concatenate((id_b[ben_idx[i]], id_m[mal_idx[i]]), 0)
    bm = np.concatenate((bm_b[ben_idx[i]], bm_m[mal_idx[i]]), 0)
    bi = np.concatenate((bi_b[ben_idx[i]], bi_m[mal_idx[i]]), 0)
    label_use = np.concatenate((ben_label[ben_idx[i]], mal_label[mal_idx[i]]), axis=0)
    
    ID = ID.reshape((-1, 1))
    bm = bm.reshape((-1, 1))
    bi = bi.reshape((-1, 1))

    print('ID', ID.shape)
    print('label', label_use.shape)

    df = pd.DataFrame(ID)
    df.to_excel(save_path + '/fold_id_'+str(i)+'.xlsx', header=['id'])

    csv = np.concatenate((ID, bm, bi, label_use), axis=1)
    name = ['id','bm','bi','shape','orien','margin','echo',
            'post_no','post_en','post_at','post_com','ca']
    df_csv = pd.DataFrame(columns=name, data=csv)
    df_csv.to_excel(save_path + '/fold_'+str(i)+'.xlsx', index=False)


print('--- done ---')