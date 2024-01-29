import pandas as pd
import numpy as np
from argparse import ArgumentParser
from pathlib import Path

class ClinicalData():
    def __init__(self, df=None):
        self.df = df

    def _gender(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _location(self, df):
        column_name = ['RUL', 'LLL', 'LUL', 'RLL', 'RML', 'right pulmonary hilar lung', '-', 'left side']
        mapping = {
            'RUL'                        : [1, 0, 0, 0, 0, 0, 0, 0],
            'right apical lung'          : [1, 0, 0, 0, 0, 0, 0, 0], # apical 頂端的
            'LLL'                        : [0, 1, 0, 0, 0, 0, 0, 0],
            'LUL'                        : [0, 0, 1, 0, 0, 0, 0, 0],
            'LLL/LUL'                    : [0, 1, 1, 0, 0, 0, 0, 0],
            'LUL/RUL/RUL'                : [1, 0, 1, 0, 0, 0, 0, 0],
            'RLL'                        : [0, 0, 0, 1, 0, 0, 0, 0],
            'LLL/RLL'                    : [0, 1, 0, 1, 0, 0, 0, 0],
            'RML'                        : [0, 0, 0, 0, 1, 0, 0, 0],
            'right pulmonary hilar lung' : [0, 0, 0, 0, 0, 1, 0, 0],
            '-'                          : [0, 0, 0, 0, 0, 0, 1, 0],
            'left side'                  : [0, 0, 0, 0, 0, 0, 0, 1],
            'RUL/RLL'                    : [1, 0, 0, 1, 0, 0, 0, 0],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _size(self, df):
        df = df.map(lambda x: x.split('/')[0])
        transformed_list = []
        for row in df.iloc[:]:
            if row == '-':
                transformed_list.append([0, 0, 0, 0, 1])
            elif float(row) >= 3:
                transformed_list.append([0, 0, 0, 1, 0])
            elif float(row) >= 2:
                transformed_list.append([0, 0, 1, 0, 0])
            elif float(row) >= 1:
                transformed_list.append([0, 1, 0, 0, 0])
            else:
                transformed_list.append([1, 0, 0, 0, 0])
        column_name = ['size<1', 'size<2', 'size<3', 'size>=3', '-']
        return pd.DataFrame(transformed_list, columns = column_name)

    def _portA_切除(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _切除肺葉(self, df):
        column_name = ['RUL', 'RML', 'RLL', 'LUL', 'LLL', 'RFL/LFL', 'LL', '-']
        mapping = {
            'RUL'     : [1, 0, 0, 0, 0, 0, 0, 0],
            'RML'     : [0, 1, 0, 0, 0, 0, 0, 0],
            'RLL'     : [0, 0, 1, 0, 0, 0, 0, 0],
            'LUL'     : [0, 0, 0, 1, 0, 0, 0, 0],
            'LLL'     : [0, 0, 0, 0, 1, 0, 0, 0],
            'RLL/RUL' : [1, 0, 1, 0, 0, 0, 0, 0],
            'RUL/RLL' : [1, 0, 1, 0, 0, 0, 0, 0],
            'RFL/LFL' : [0, 0, 0, 0, 0, 1, 0, 0],
            'LL'      : [0, 0, 0, 0, 0, 0, 1, 0],
            '-'       : [0, 0, 0, 0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _Differentiation分化(self, df):
        df = df.reset_index(drop=True)
        df = df.str.capitalize()
        return pd.get_dummies(df)

    def _LVI(self, df):
        column_name = ['Present', 'Not Present', 'Not identified', '-']
        mapping = {
            # same
            'Present'              : [1, 0, 0, 0],
            'Present, diffuse'     : [1, 0, 0, 0],
            'yes'                  : [1, 0, 0, 0],
            # same
            'Not Present'          : [0, 1, 0, 0],
            'Not Present '         : [0, 1, 0, 0],
            # same
            'Not identified'       : [0, 0, 1, 0],
            'Cannot be determined' : [0, 0, 1, 0],
            'Indeterminate'        : [0, 0, 1, 0],
            '-'                    : [0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _ALK(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _ROS(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _EGFR(self, df):
        column_name = ['EGFR(+)', 'EGFR(-)', '-']
        mapping = {
            # same
            'EGFR(+)' : [1, 0, 0],
            'EFGR(+)' : [1, 0, 0],
            # same
            'EGFR(-)' : [0, 1, 0],
            'EFGR(-)' : [0, 1, 0],
            '-'       : [0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _EGFR_mutation(self, df):
        column_name = ['L858R', 'T790M', 'Exon 19', 'Exon 20', 'Exon 21', 'L861Q',
                       'HER2 Exon 20 insert', '-']
        mapping = {
            'L858R'               : [1, 0, 0, 0, 0, 0, 0, 0],
            'T790M'               : [0, 1, 0, 0, 0, 0, 0, 0],
            'L858R/T790M'         : [1, 1, 0, 0, 0, 0, 0, 0],
            'Exon 19'             : [0, 0, 1, 0, 0, 0, 0, 0],
            'Exon 20'             : [0, 0, 0, 1, 0, 0, 0, 0],
            'Exon 21'             : [0, 0, 0, 0, 1, 0, 0, 0],
            'L861Q'               : [0, 0, 0, 0, 0, 1, 0, 0],
            'HER2 Exon 20 insert' : [0, 0, 0, 0, 0, 0, 1, 0],
            '-'                   : [0, 0, 0, 0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _臨床分期T(self, df):
        column_name = ['1', '2', '3', '4', 'x']
        mapping = {
            '1'   : [1, 0, 0, 0, 0],
            '1a'  : [1, 0, 0, 0, 0],
            '1b'  : [1, 0, 0, 0, 0],
            '1b2' : [1, 0, 0, 0, 0],
            '1c'  : [1, 0, 0, 0, 0],
            '1mi' : [1, 0, 0, 0, 0],
            '2'   : [0, 1, 0, 0, 0],
            '2a'  : [0, 1, 0, 0, 0],
            '2b'  : [0, 1, 0, 0, 0],
            '3'   : [0, 0, 1, 0, 0],
            '4'   : [0, 0, 0, 1, 0],
            'x'   : [0, 0, 0, 0, 1],
            '-'   : [0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _臨床分期N(self, df):
        column_name = ['0', '1', '2', '3', 'x']
        mapping = {
            '0' : [1, 0, 0, 0, 0],
            '1' : [0, 1, 0, 0, 0],
            '2' : [0, 0, 1, 0, 0],
            '3' : [0, 0, 0, 1, 0],
            'x' : [0, 0, 0, 0, 1],
            '-' : [0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _臨床分期M(self, df):
        column_name = ['0', '1', 'x']
        mapping = {
            '0'  : [1, 0, 0],
            '1'  : [0, 1, 0],
            '1a' : [0, 1, 0],
            '1b' : [0, 1, 0],
            '1c' : [0, 1, 0],
            'x'  : [0, 0, 1],
            '-'  : [0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _臨床分期stage(self, df):
        column_name = [
            'I', 'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB', 'IIIA', 'IIIB', 'IIIC',
            'IV', 'IVA', 'IVB', '-'
        ]
        mapping = {
            #1
            'I'    : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IA1'  : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IA2'  : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IA3'  : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IB'   : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #2
            'IIA'  : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'IIB'  : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            #3
            'IIIA' : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'IIIB' : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            '3B'   : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'IIIC' : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            #4
            'IV'   : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'IVA'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'IVB'  : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            #same
            '-'    : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # - - - -
            'C'    : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # - - - C
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _病理分期T(self, df):
        column_name = ['1', '2', '3', '4', 'X']
        mapping = {
            '1'   : [1, 0, 0, 0, 0],
            '1a'  : [1, 0, 0, 0, 0],
            '1b'  : [1, 0, 0, 0, 0],
            '1c'  : [1, 0, 0, 0, 0],
            '1C'  : [1, 0, 0, 0, 0],
            '1mi' : [1, 0, 0, 0, 0],
            '2'   : [0, 1, 0, 0, 0],
            '2a'  : [0, 1, 0, 0, 0],
            '2b'  : [0, 1, 0, 0, 0],
            '3'   : [0, 0, 1, 0, 0],
            '4'   : [0, 0, 0, 1, 0],
            'X'   : [0, 0, 0, 0, 1],
            '-'   : [0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _病理分期N(self, df):
        column_name = ['0', '1', '2', '3', 'x']
        mapping = {
            '0'    : [1, 0, 0, 0, 0],
            '1'    : [0, 1, 0, 0, 0],
            '2'    : [0, 0, 1, 0, 0],
            '3'    : [0, 0, 0, 1, 0],
            '1mic' : [0, 1, 0, 0, 0],
            'x'    : [0, 0, 0, 0, 1],
            '-'    : [0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _病理分期M(self, df):
        column_name = ['0', '1', '-']
        mapping = {
            '0'  :[1, 0, 0],
            '1a' :[0, 1, 0],
            '1b' :[0, 1, 0],
            '1c' :[0, 1, 0],
            '-'  :[0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _病理分期stage(self, df):
        column_name = [
            'IA1', 'IA2', 'IA3', 'IB', 'IIA', 'IIB', 'IIIA', 'IVA', 'IVB', '-'
        ]
        mapping = {
            'IA1'  : [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'IA2'  : [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'IA3'  : [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'IB'   : [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'IIA'  : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'IIB'  : [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'IIIA' : [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'IVA'  : [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'IVB'  : [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            '-'    : [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        }
        transformed_list = [mapping[row] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _有無轉移(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _轉移器官(self, df):
        淋巴腺 = [
            'paraaortic lymph node',
            'lymph nodes',
            'Lymph node',
            'subcarinal',
            'neck lymph node',
            'LN',
            'lymph node',
            'gastric lymph node',
            'mediastinal lymph nodes',
            'lower neck nodal',
            'mediastinum lymph nodes',
        ]
        腦部 = [
            'brain',
            'contralateral lobe', # 對側葉
            'cerebellum', # 小腦
            'cerebellar',
            'Brain',
        ]
        骨骼 = [
            'skull',
            'bone',
            'Bone',
            'spine',
            'spinal',
            'vertebral',
            'bones',
        ]
        肝臟 = [
            'liver',
        ]
        腎上腺 = [
            'adrenal', # 腎上腺
            'adrenal gland',
            'adrenal ',
        ]
        其他 = [
            'right lower paratracheal space',
            'scapular',
            'pancreas',
            'kidney',
            'axillary',
            'lung adeocarcinoma',
            'right adrenal metastases',
            'Ipsilateral hilar',
            'buccal',
            'Lung',
            'extrathoracic',
            'gastric',
            'Malignant pleural effusion',
            'sacroiliac joint',
            'peritoneal',
            'spleen',
            'HCC',
            'Kidney',
            'Pleura',
            'Pleural seeding',
            'acetabulum',
            'lung',
            'femur',
            'peribronchial node',
            'pericardial',
            'pleural',
            'pulmonary',
            'pretracheal',
            'neck',
            'pleura'
        ]
        無 = [
            '-',
        ]
        transformed_list = []
        for row in df.iloc[:]:
            轉移器官s = row.split(', ')
            a_patient = [0, 0, 0, 0, 0, 0, 0]
            a_patient[0] = 1 if set(淋巴腺).intersection(轉移器官s) else 0
            a_patient[1] = 1 if set(腦部).intersection(轉移器官s) else 0
            a_patient[2] = 1 if set(骨骼).intersection(轉移器官s) else 0
            a_patient[3] = 1 if set(肝臟).intersection(轉移器官s) else 0
            a_patient[4] = 1 if set(腎上腺).intersection(轉移器官s) else 0
            a_patient[5] = 1 if set(其他).intersection(轉移器官s) else 0
            a_patient[6] = 1 if set(無).intersection(轉移器官s) else 0
            transformed_list.append(a_patient)
        column_name = ['淋巴腺轉移', '腦部轉移', '骨骼轉移', '肝臟轉移', '腎上腺轉移', '其他轉移', '無轉移']
        return pd.DataFrame(transformed_list, columns = column_name)

    def _有無復發(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _復發器官(self, df):
        肝臟 = [
            'liver'
        ]
        淋巴腺 = [
            'lymph node',
            'mediastinal lymph nodes',
            'thoracic lymphadenopathy', # 淋巴腺腫大
        ]
        肺 = [
            'obstructive pneumonitis', # 肺炎腫
            'lung',
        ]
        腦部 = [
            'cerebellum',
            'intracranial', # 顱內
            'brain',
        ]
        骨骼 = [
            'bone',
        ]
        其他 = [
            'thrombocytopenia', # 血小板
            'diaphragm', # 隔膜
            'pleural', # 胸膜
            'gastric', # 胃
        ]
        無 = [
            '-',
        ]
        transformed_list = []
        for row in df.iloc[:]:
            復發器官s = row.split(', ')
            a_patient = [0, 0, 0, 0, 0, 0, 0]
            a_patient[0] = 1 if set(肝臟).intersection(復發器官s) else 0
            a_patient[1] = 1 if set(淋巴腺).intersection(復發器官s) else 0
            a_patient[2] = 1 if set(肺).intersection(復發器官s) else 0
            a_patient[3] = 1 if set(腦部).intersection(復發器官s) else 0
            a_patient[4] = 1 if set(骨骼).intersection(復發器官s) else 0
            a_patient[5] = 1 if set(其他).intersection(復發器官s) else 0
            a_patient[6] = 1 if set(無).intersection(復發器官s) else 0
            transformed_list.append(a_patient)
        column_name = ['肝臟復發', '淋巴腺復發', '肺復發', '腦部復發', '骨骼復發', '其他復發', '無復發']
        return pd.DataFrame(transformed_list, columns = column_name)
        
    def _Smoking(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _PPD(self, df):
        transformed_list = []
        for row in df.iloc[:]:
            if row == '-':
                transformed_list.append([0, 0, 0, 0, 1])
            elif float(row) >= 3:
                transformed_list.append([0, 0, 0, 1, 0])
            elif float(row) >= 2:
                transformed_list.append([0, 0, 1, 0, 0])
            elif float(row) >= 1:
                transformed_list.append([0, 1, 0, 0, 0])
            else:
                transformed_list.append([1, 0, 0, 0, 0])
        column_name = ['PPD<1', 'PPD<2', 'PPD<3', 'PPD>=3', '-']
        return pd.DataFrame(transformed_list, columns = column_name)

    def _HTN_高血壓(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _DM_糖尿病(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _存在lung_cancer家族病史(self, df):
        df = df.reset_index(drop=True)
        return pd.get_dummies(df)

    def _Complications併發症(self, df):
        column_name = ['併發症']
        transformed_list = [[1] if row != '-' else [0] for row in df.iloc[:]]
        return pd.DataFrame(transformed_list, columns = column_name)

    def _FVC_Predicted(self, df):
        transformed_list = []
        for row in df.iloc[:]:
            if row == '-':
                transformed_list.append([0, 0, 1])
            elif float(row) < 80:
                transformed_list.append([1, 0, 0])
            else:
                transformed_list.append([0, 1, 0])
        column_name = ['FVC_%Predicted<0.8', 'FVC_%Predicted>=0.8', '-']
        return pd.DataFrame(transformed_list, columns = column_name)

    def _FEV1_Predicted(self, df):
        transformed_list = []
        for row in df.iloc[:]:
            if row == '-':
                transformed_list.append([0, 0, 1])
            elif float(row) < 80:
                transformed_list.append([1, 0, 0])
            else:
                transformed_list.append([0, 1, 0])
        column_name = ['FEV1_%Predicted<0.8', 'FEV1_%Predicted>=0.8', '-']
        return pd.DataFrame(transformed_list, columns = column_name)

    def _Observed(self, df):
        transformed_list = []
        for row in df.iloc[:]:
            if row == '-':
                transformed_list.append([0, 0, 1])
            elif float(row) < 70:
                transformed_list.append([1, 0, 0])
            else:
                transformed_list.append([0, 1, 0])
        column_name = ['FEV1/FVC<0.7', 'FEV1/FVC>=0.7', '-']
        return pd.DataFrame(transformed_list, columns = column_name)
    
    def _voi_size(self, df):
        df = df.astype(int)
        df['x_range'] = df['x_end'] - df['x_start'] + 1
        df['y_range'] = df['y_end'] - df['y_start'] + 1
        df['z_range'] = df['z_end'] - df['z_start'] + 1
        df = df.drop(columns=['x_start', 'y_start', 'x_end', 'y_end', 'z_start', 'z_end'])
        return df.reset_index(drop=True)

    def Preprocess(self):
        df_gender = self._gender(self.df['Sex'])
        df_location = self._location(self.df['Location'])
        df_size = self._size(self.df['Size(cm)'])
        df_portA_切除 = self._portA_切除(self.df['PORT-A/切除'])
        df_切除肺葉 = self._切除肺葉(self.df['切除肺葉'])
        df_Differentiation分化 = self._Differentiation分化( self.df['Differentiation分化'])
        df_LVI = self._LVI(self.df['LVI(Lymphovascular invasion)'])
        df_ALK = self._ALK(self.df['ALK'])
        df_ROS = self._ROS(self.df['ROS-1'])
        df_EGFR = self._EGFR(self.df['EGFR'])
        df_EGFR_mutation = self._EGFR_mutation(self.df['EGFR mutation'])
        df_臨床分期T = self._臨床分期T(self.df['臨床分期T'])
        df_臨床分期N = self._臨床分期N(self.df['臨床分期N'])
        df_臨床分期M = self._臨床分期M(self.df['臨床分期M'])
        df_臨床分期stage = self._臨床分期stage(self.df['臨床分期stage'])
        df_病理分期T = self._病理分期T(self.df['病理分期T'])
        df_病理分期N = self._病理分期N(self.df['病理分期N'])
        df_病理分期M = self._病理分期M(self.df['病理分期M'])
        df_病理分期stage = self._病理分期stage(self.df['病理分期stage'])
        df_有無轉移 = self._有無轉移(self.df['有無轉移'])
        df_轉移器官 = self._轉移器官(self.df['轉移器官'])
        df_有無復發 = self._有無復發(self.df['有無復發'])
        df_復發器官 = self._復發器官(self.df['復發器官'])
        df_Smoking = self._Smoking(self.df['Smoking'])
        df_ppd = self._PPD(self.df['PPD'])
        df_HTN_高血壓 = self._HTN_高血壓(self.df['HTN 高血壓'])
        df_DM_糖尿病 = self._DM_糖尿病(self.df['DM 糖尿病'])
        df_存在lung_cancer家族病史 = self._存在lung_cancer家族病史(self.df['存在lung cancer家族病史'])
        df_Complications併發症 = self._Complications併發症(self.df['Complications併發症'])
        df_FVC_Predicted = self._FVC_Predicted(self.df['FVC_%Predicted'])
        df_FEV1_Predicted = self._FEV1_Predicted(self.df['FEV1_%Predicted'])
        df_observed = self._Observed(self.df['Observed'])
        df_voi_size = self._voi_size(self.df[['x_start', 'y_start', 'x_end', 'y_end', 'z_start', 'z_end']])
        new_df = pd.concat([
            self.df['Patient ID'].reset_index(drop=True),
            self.df['是否存活'].reset_index(drop=True),
            df_gender, 
            df_location,
            df_size,
            df_portA_切除,
            df_切除肺葉,
            df_Differentiation分化,
            df_LVI,
            df_ALK,
            df_ROS,
            df_EGFR,
            df_EGFR_mutation,
            df_臨床分期T,
            df_臨床分期N,
            df_臨床分期M,
            df_臨床分期stage,
            df_病理分期T,
            df_病理分期N,
            df_病理分期M,
            df_病理分期stage,
            df_有無轉移,
            df_轉移器官,
            df_有無復發,
            df_復發器官,
            df_Smoking,
            df_ppd,
            df_HTN_高血壓,
            df_DM_糖尿病,
            df_存在lung_cancer家族病史,
            df_Complications併發症,
            df_FVC_Predicted,
            df_FEV1_Predicted,
            df_observed,
            df_voi_size,
        ], axis=1)
        print(len(new_df.columns)) # 151

        cols = []
        count = 1
        for column in new_df.columns:
            if column == '-':
                cols.append(f'{count}_-')
                count+=1
            else:
                cols.append(column)
        new_df.columns = cols

        return new_df

def main(args):
    df = pd.read_excel(args.input_path, sheet_name=0, dtype = str)
    # change header
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    clinical_data = ClinicalData(df)
    new_df = clinical_data.Preprocess()
    new_df.to_excel(args.output_path, index=False) 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=Path, default='../../Lung GSI patient list_20220818_查資料.xlsx')
    parser.add_argument('-o', '--output_path', type=Path, default='../../transformed_clinical_data.xlsx')
    args = parser.parse_args()
    main(args)