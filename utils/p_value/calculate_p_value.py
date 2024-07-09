import pandas as pd
import numpy as np
from mlxtend.evaluate import mcnemar
from mlxtend.evaluate import mcnemar_table
from Delong import *

target_csv =pd.read_csv(r"D:\Users\tsuyi\Desktop\NTU_LAB402_research_code\utils\p_value\data\90d_DualWay.csv")
label=np.array(target_csv['label'])
y_target=np.array(target_csv['prob'])


compare_csv=pd.read_csv(r"D:\Users\tsuyi\Desktop\NTU_LAB402_research_code\utils\p_value\data\Kaplan-Meier-90.csv")
y_cmp=np.array(compare_csv['prob'])
label2 = np.array(compare_csv['label'])

#for i,_ in enumerate(label):
#    if label[i]!= label2[i]:
#        assert 'a'=='b'
# print(y_target, y_cmp)
DelongTest(y_target, y_cmp, label)

y_target[y_target>0.5]=1
y_target[y_target<=0.5]=0
y_cmp[y_cmp>0.5]=1
y_cmp[y_cmp<=0.5]=0

# print(y_target, y_cmp)
tb = mcnemar_table(y_target=label,y_model1=y_target,y_model2=y_cmp)
chi2, p = mcnemar(ary=tb, corrected=True)
print("Mcnemar:", p)




