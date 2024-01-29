import os
import csv
import torch
import warnings
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix, roc_curve, auc
# model library
from ConvNeXt import convnext_tiny
# dataset library
from dataset import Lung_Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')
    
def test(args, k: int, device, model, testloader):

    #test stage#
    test_label, test_pred, test_probability = [], [], []

    print("k = ", k)

    model.eval()
    with torch.no_grad():
        for _, img, clinical_info, voi_size, label in testloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)
            output = model(img, clinical_info, voi_size)  #[bs,o)
            # _, pred = torch.max(output, dim=1) # label = 0 的機率大於 0.5 則本次預設結果為 0
            pred = torch.where(output[:, 0] > 0.475, 0, 1) # label = 0 的機率大於 0.475 就算本次預設結果為 0
            test_label.append((label.detach().cpu().numpy())[0])
            test_pred.append((pred.detach().cpu().numpy())[0])

            pred = output[:, 0] # 計算 auc 用
            test_probability.append((pred.detach().cpu().numpy())[0])

        tp, fn, fp, tn = confusion_matrix(test_label, test_pred).ravel()
        print(tp, fn, fp, tn)
        tnr, tpr, ppv, npv = tn/(fp+tn), tp/(tp+fn), tp/(tp+fp), tn/(tn+fn)
        accuracy = (tn + tp) / (tn + fp + fn + tp)
        
        Fpr, Tpr, _ = roc_curve(test_label, test_probability, pos_label=0)
        Auc = auc(Fpr, Tpr)
        print(f"test dataset")
        print(f"accuracy = {accuracy:.2}, tnr = {tnr:.2}, tpr = {tpr:.2}, ppv = {ppv:.2}, npv = {npv:.2}, auc = {Auc:.2}\n")
         
    return accuracy, tnr, tpr, ppv, npv, Auc

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
    parser.add_argument(
        "--model_dir",
        type=Path,
        default='./model'
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default='./3year_result/'
    )

    parser.add_argument(
        "--nfold",
        help="number of fold (cross validation)",
        type=int,
        default=5
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args

if __name__ == '__main__' :
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    metrics = {
        'acc' : np.zeros(args.nfold),
        'tnr' : np.zeros(args.nfold),
        'tpr' : np.zeros(args.nfold),
        'ppv' : np.zeros(args.nfold),
        'npv' : np.zeros(args.nfold),
        'auc' : np.zeros(args.nfold),
    }

    for k in range(1, args.nfold+1):

        # model
        model = convnext_tiny(len_of_clinical_features=150)
        model_path = args.model_dir / Path('tradeoff_'+str(k)).with_suffix('.pth')
        # model_path = args.model_dir / Path(str(k)).with_suffix('.pth')
        if model_path.exists():
            check = torch.load(model_path)
            print('model exist')
            model.load_state_dict(check['model_state_dict'])
        else:
            raise('model doesn\'t exist')
        model.to(device)

        testset = Lung_Dataset(
            images_path=args.input_dir,
            clinical_data_path=args.clinical_data_path,
            train_val_test_list_path=args.train_val_test_path,
            mode="test", 
            k=k,
            predict_mode=True,
        )
        testloader = DataLoader(testset, batch_size=1, shuffle=False)      

        metric = test(args, k, device, model, testloader)

        for key, m in zip(metrics.keys(), metric):
            metrics[key][k-1] = m

    print(f"accuracy = {np.mean(metrics['acc']):.4}±{np.std(metrics['acc'], ddof=1):.4},\
            SEN = {np.mean(metrics['tpr']):.4}±{np.std(metrics['tpr'], ddof=1):.4}, \
            SPEC = {np.mean(metrics['tnr']):.4}±{np.std(metrics['tnr'], ddof=1):.4}, \
            auc = {np.mean(metrics['auc']):.4}±{np.std(metrics['auc'], ddof=1):.4}")

    result_path = args.output_dir / Path('fivefold3.csv')
    with open(result_path, "a+", newline="") as f:         
        fieldnames = ['acc', 'tnr' ,'tpr' , 'ppv', 'npv', 'auc', 
                      'std_acc', 'std_tnr', 'std_tpr', 'std_ppv', 'std_npv', 'std_auc',
                      'Name']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                'acc': np.mean(metrics['acc']), 
                'tnr': np.mean(metrics['tnr']), 
                'tpr': np.mean(metrics['tpr']), 
                'ppv': np.mean(metrics['ppv']), 
                'npv': np.mean(metrics['npv']), 
                'auc': np.mean(metrics['auc']), 
                'std_acc': np.std(metrics['acc'], ddof=1), 
                'std_tnr': np.std(metrics['tnr'], ddof=1), 
                'std_tpr': np.std(metrics['tpr'], ddof=1), 
                'std_ppv': np.std(metrics['ppv'], ddof=1), 
                'std_npv': np.std(metrics['npv'], ddof=1), 
                'std_auc': np.std(metrics['auc'], ddof=1),
                'Name': args.model_dir.stem
            }
        )