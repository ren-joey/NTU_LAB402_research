import torch
import torch.utils.data
from torch.utils.data import DataLoader
import os
import numpy as np
import warnings
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
# from statsmodels.stats.contingency_tables import mcnemar
# model library
from ResNeSt50 import ResNeSt50
from ConvNeXt import convnext_tiny
# dataset library
from dataset import Lung_Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')
    
def test(args, k: int, device, model1, model2, testloader):

    #test stage#
    test_pred1, test_pred2, test_label1, test_label2 = [], [], [], []
    test_names1, test_names2 = [], []

    print("k = ", k)

    model1.eval()
    with torch.no_grad():
        for name, img, clinical_info, voi_size, label in testloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)
            
            output = model1(img, clinical_info, voi_size)  #[bs,o)
            pred = torch.where(output[:, 0] > 0.475, 0, 1)
            test_pred1.append((pred.detach().cpu().numpy())[0])
            test_label1.append((label.detach().cpu().numpy())[0])
            test_names1.append(name)

    model2.eval()
    with torch.no_grad():
        for name, img, clinical_info, voi_size, label in testloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)

            output = model2(img, clinical_info, voi_size)  #[bs,o)
            pred = torch.where(output[:, 0] > 0.475, 0, 1)
            test_pred2.append((pred.detach().cpu().numpy())[0])
            test_label2.append((label.detach().cpu().numpy())[0])
            test_names2.append(name)

    assert test_names1 == test_names2, 'ID 排序必須要相等'
    assert test_label1 == test_label2, 'label 排序必須要相等'

    return test_pred1, test_pred2, test_label1, test_label2, test_names1, test_names2

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
        default='./model/Other Models/ResNeSt'
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

if __name__ == '__main__' :
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    test_pred1, test_pred2, test_label, test_names = [], [], [], []

    for k in range(1, args.nfold+1):

        model1 = ResNeSt50()
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

        tmp1, tmp2, _, tmp_label, _, tmp_names = test(args, k, device, model1, model2, testloader)
        test_pred1 += tmp1
        test_pred2 += tmp2
        test_label += tmp_label
        test_names += tmp_names
    test_pred1, test_pred2, test_label = np.array(test_pred1), np.array(test_pred2), np.array(test_label)
    tb = mcnemar_table( y_target=test_label, 
                        y_model1=test_pred1, 
                        y_model2=test_pred2 )

    print('p-value:', mcnemar(ary=tb, corrected=True))