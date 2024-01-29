import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import numpy as np
import warnings
from argparse import ArgumentParser
from pathlib import Path
import scipy.stats as st
from scipy.stats import norm
from sklearn.metrics import roc_curve, auc

# model library
from ConvNeXt import convnext_tiny
from ResNet50 import ResNet50 
# dataset library
from dataset import Lung_Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')
    
def test(args, k: int, device, model, testloader):

    #test stage#
    test_pred, test_label= [], []
    test_names = []

    print("k = ", k)

    model.eval()
    with torch.no_grad():
        for name, img, clinical_info, voi_size, label in testloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)
            
            output = model(img, clinical_info, voi_size)
            label = label.detach().cpu().numpy()[0]
            pred = output[:, 0].detach().cpu().numpy()[0]

            test_pred.append(pred)
            test_label.append(label)
            test_names.append(name)

    return test_pred, test_label, test_names

def arg_parse():
    parser = ArgumentParser()
    parser.add_argument(
        "--clinical_data_path",
        type=Path,
        default='../transformed_clinical_data.xlsx', 
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default='../../data/C+_140/voi_128_224_224',
    )
    parser.add_argument(
        "--train_val_test_path",
        type=Path,
        default='../train_val_test_split.xlsx'
    )
    # model1 的路徑 (要比較的 model)
    parser.add_argument(
        "--model1_dir",
        type=Path,
        default='./model/Other Models/ResNet'
    )
    # model2 的路徑 (最好的 model)
    parser.add_argument(
        "--model2_dir",
        type=Path,
        default='./model/Dual Energy CT with different keV/C+140'
    )

    parser.add_argument(
        "--nfold",
        help="number of fold (cross validation)",
        type=int,
        default=5
    )

    args = parser.parse_args()
    return args

def delong_test(test_label1, test_pred1, test_label2, test_pred2):
    # Calculate the AUC values for the ROC curves
    fpr, tpr, _ = roc_curve(test_label1, test_pred1, pos_label=0)
    model1_auc = round(auc(fpr, tpr), 4)
    print('model1_auc = ', model1_auc)
    fpr, tpr, _ = roc_curve(test_label2, test_pred2, pos_label=0)
    model2_auc = round(auc(fpr, tpr), 4)
    print('model2_auc = ', model2_auc)

    # Perform the DeLong test
    n1 = len(test_pred1)
    n2 = len(test_pred2)

    test_pred1 = np.asarray(test_pred1)
    test_pred2 = np.asarray(test_pred2)
    variance = np.var(test_pred1) / n1 + np.var(test_pred2) / n2
    z_value = (model1_auc - model2_auc) / np.sqrt(variance)
    p_value = 2 * (1 - norm.cdf(abs(z_value)))

    print(f"The p-value is: {p_value}")
    
class DelongTest():
    def __init__(self, preds1, preds2, label, threshold =0.05):
        self._preds1 = preds1
        self._preds2 = preds2
        self._label = label
        self.threshold = threshold
        self._show_result()

    def _auc(self, X,Y):
        return 1/(len(X)*len(Y))*sum([self._kernel(x,y) for x in X for y in Y])
    
    def _kernel(self, X, Y) -> float:
        return 0.5 if Y==X else int(Y<X)
    
    def _structural_components(self, X,Y) :
        V10 = [1/len(Y)*sum([self._kernel(x,y) for y in Y ]) for x in X]
        V01 = [1/len(X)*sum([self._kernel(x,y) for x in X ]) for y in Y]
        return V10, V01

    def _get_S_entry(self, V_A, V_B, auc_A, auc_B):
        return 1/(len(V_A)-1)*sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
    
    def _z_score(self,var_A, var_B, covar_AB, auc_A, auc_B):
        return (auc_A- auc_B)/((var_A+ var_B- 2*covar_AB)**(0.5)+1e-8)
    
    def _group_preds_by_label(sefl,preds,actual):
        # print(preds)
        # print(lactualabels)
        X = [p for (p,a) in zip(preds, actual) if a]
        Y = [p for (p,a) in zip(preds, actual) if not a]
        return X, Y

    def _compute_z_p(self):
        X_A, Y_A = self._group_preds_by_label(self._preds1, self._label)
        X_B, Y_B = self._group_preds_by_label(self._preds2, self._label)

        V_A10, V_A01 = self._structural_components(X_A, Y_A)
        V_B10, V_B01 = self._structural_components(X_B, Y_B)
        auc_A = self._auc(X_A, Y_A)
        auc_B = self._auc(X_B, Y_B)

        var_A = (self._get_S_entry(V_A10, V_A10,auc_A, auc_A)*1/len(V_A10)+ self._get_S_entry(V_A01, V_A01, auc_A, auc_A)*1/len(V_A01))
        var_B = (self._get_S_entry(V_B10, V_B10,auc_B, auc_B)*1/len(V_B10)+ self._get_S_entry(V_B01, V_B01, auc_B, auc_B)*1/len(V_B01))

        covar_AB = (self._get_S_entry(V_A10, V_B10,auc_A, auc_B)*1/len(V_A10)+self._get_S_entry(V_A01, V_B01,auc_A, auc_B)*1/len(V_A01))

        z = self._z_score(var_A, var_B, covar_AB, auc_A, auc_B)
        p = st.norm.sf(abs(z))*2

        return z,p
    
    def _show_result(self):
        z, p = self._compute_z_p()
        print(f"Delong's p value= {p:.5f};")

        if p<self.threshold:
            print("There is a significant difference.")
        else:
            print("There is no significant difference.")
    
if __name__ == '__main__' :
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    test_pred1, test_pred2, test_label1, test_label2, test_names1, test_names2 = [], [], [], [], [], []

    for k in range(1, args.nfold+1):
        # model
        model1 = ResNet50()
        model1_path = args.model1_dir / Path('tradeoff_'+str(k)).with_suffix('.pth')
        model1.load_state_dict(torch.load(model1_path)['model_state_dict'])
        model1.to(device)

        model2 = convnext_tiny()
        model2_path = args.model2_dir / Path('tradeoff_'+str(k)).with_suffix('.pth')
        model2.load_state_dict(torch.load(model2_path)['model_state_dict'])
        model2.to(device)

        testset = Lung_Dataset(
            images_path=args.input_dir,
            clinical_data_path=args.clinical_data_path,
            train_val_test_list_path=args.train_val_test_path,
            mode="test", 
            k=k,
            predict_mode=True,
        )
        testloader = DataLoader(testset, batch_size=1, shuffle=False)

        tmp, tmp_label, tmp_names = test(args, k, device, model1, testloader)
        test_pred1 += tmp
        test_label1 += tmp_label
        test_names1 += tmp_names

        tmp, tmp_label, tmp_names = test(args, k, device, model2, testloader)
        test_pred2 += tmp
        test_label2 += tmp_label
        test_names2 += tmp_names

    assert test_names1 == test_names2, 'ID 排序必須要相等'
    assert test_label1 == test_label2, 'label 排序必須要相等'

    # 算 delong's test
    delong_test(test_label1, test_pred1, test_label2, test_pred2)

    # 另一個 code，也是算 delong's test
    DelongTest(test_pred1, test_pred2, test_label1)