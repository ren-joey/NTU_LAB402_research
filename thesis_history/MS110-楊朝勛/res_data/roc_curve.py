import os
import warnings
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import matplotlib.pyplot as plt
# model library
from ConvNeXt import convnext_tiny
# dataset library
from dataset import Lung_Dataset

os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')
    
def test(args, k: int, device, model, testloader):

    #test stage#
    pred0, pred1 = [], []

    print("k = ", k)

    model.eval()
    with torch.no_grad():
        for name, img, clinical_info, voi_size, label in testloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)
            
            output = model(img, clinical_info, voi_size)  #[bs,o)
            label = label.squeeze()

            pred = output[:, 1]
            pred = (pred - 0.525) * 0.5 / 0.475 + 0.5
            pred = pred.detach().cpu().numpy()[0]

            label = label.detach().cpu().numpy()
            if label == 0:
                pred0.append(pred)
            else:
                pred1.append(pred)

    return pred0, pred1

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

    #set up plotting area
    plt.figure(0).clf()

    # 可以再這邊一次放入多個模型
    model_paths = {'./model/Dual Energy CT with different keV/C+140': 'C+140'}

    for path, path_name in model_paths.items():
        print('model path = ', path)
        pred0, pred1 = [], []
        for k in range(1, args.nfold+1):
            model = convnext_tiny(len_of_clinical_features=150)
            model_path = path / Path('tradeoff_'+str(k)).with_suffix('.pth')
            model.load_state_dict(torch.load(model_path)['model_state_dict'])
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

            tmp_pred0, tmp_pred1 = test(args, k, device, model, testloader)

            pred0 += tmp_pred0
            pred1 += tmp_pred1

        _path = path_name + ".txt"
        f = open(_path, 'w')
        f.write('method 1'+'\n')
        f.write('Large'+'\n'+'\n')
        for ben in pred0:
            f.write(str(np.array([ben]))[1:-1]+'\n')
        f.write('*\n')
        for mali in pred1:
            f.write(str(np.array([mali]))[1:-1]+'\n')
        f.write('*\n')
        f.close()