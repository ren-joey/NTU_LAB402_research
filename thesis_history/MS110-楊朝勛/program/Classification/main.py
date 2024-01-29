import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
from typing import Dict
from pathlib import Path
import random
from argparse import ArgumentParser
# model library
from ConvNeXt import convnext_tiny
# dataset library
from dataset import Lung_Dataset
# loss library
from torch_focal_loss import FocalLoss
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
os.environ['KMP_DUPLICATE_LIB_OK']='True'
warnings.filterwarnings('ignore')

def save_model(model_dir, model, args, epoch=None):
    torch.save({
        'epochs':args.epochs,
        'model_state_dict':model.state_dict(),
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'patience': args.patience,
    }, model_dir)

def get_label_weights(args, k):
    weights = [3, 1]
    print('weights = ', weights)
    weights = torch.tensor(weights.copy(), dtype=torch.float32)
    weights = weights.to(device)
    return weights

def train(args, k: int, device, model, optimizer, criterion, scheduler, trainloader, valloader, seed):
    
    the_last_loss = 100
    trigger_times = 0

    the_best_loss = 0
    the_best_accuracy = 0

    save_path = args.model_dir / Path(str(k)).with_suffix('.pth')
    save_tradoff_path = args.model_dir / Path('tradeoff_'+str(k)).with_suffix('.pth')

    for epoch in range(args.epochs):
        epoch_loss, epoch_accuracy = 0, 0
        train_label, train_pred = [], []
        model.train()
        for i, (_, img, clinical_info, voi_size, label) in enumerate(trainloader):
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)

            output = model(img, clinical_info, voi_size)  #[bs, 2]
            torch.cuda.empty_cache()

            # _, pred = torch.max(output, dim=1)
            pred = torch.where(output[:, 0] > 0.475, 0, 1)
            loss = criterion(output, label)  #focal
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().item()
            epoch_accuracy += torch.sum(pred == label)

            train_label += label.detach().cpu().numpy().tolist()
            train_pred += pred.detach().cpu().numpy().tolist()

        scheduler.step()

        epoch_loss = epoch_loss / (len(trainloader) * args.batch_size)
        epoch_accuracy = epoch_accuracy.float() / (len(trainloader) * args.batch_size)
        tp, fn, fp, tn = confusion_matrix(train_label, train_pred).ravel()
        tnr, fpr, fnr, tpr = tn/(fp+tn), fp/(fp+tn), fn/(tp+fn), tp/(tp+fn)
        print(f'[{k}/{5}][{epoch}/{args.epochs}] loss: {epoch_loss:.8}, accuracy: {epoch_accuracy:.2}, tnr = {tnr:.2}, tpr = {tpr:.2}')

        # Early stopping
        the_current_loss, the_current_accuracy, the_current_tnr, the_current_tpr = validation(device, model, criterion, valloader)
        print(f'[validation] The current loss: {the_current_loss:.8}, accuracy: {the_current_accuracy:.2}, tnr = {the_current_tnr:.2}, tpr = {the_current_tpr:.2}')

        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)

            if trigger_times >= args.patience:
                print(f'Early stopping!\nEpoch = {epoch}')
                return
        else:
            print('trigger times: 0')
            trigger_times = 0

        if epoch == 0 or the_best_loss >= the_current_loss:
            print('Recording best model.')
            save_model(save_path, model, args, epoch)
            the_best_loss = the_current_loss

        if epoch == 0 or (the_best_accuracy <= the_current_accuracy and abs(the_current_tnr-the_current_tpr) <= 0.15):
            print('Recording best tradeoff model.')
            save_model(save_tradoff_path, model, args, epoch)
            the_best_accuracy = the_current_accuracy

        the_last_loss = the_current_loss

    print(f'stopping! Epoch = {args.epochs}')
    return 

def validation(device, model, criterion, valloader):
    model.eval()
    val_loss = 0
    val_accuracy = 0
    val_label, val_pred = [], []

    with torch.no_grad():
        for _, img, clinical_info, voi_size, label in valloader:
            img, clinical_info, voi_size, label = img.to(device), clinical_info.to(device), voi_size.to(device), label.to(device)
            output = model(img, clinical_info, voi_size)  #[bs, 2]
            # _, pred = torch.max(output, dim=1)
            pred = torch.where(output[:, 0] > 0.475, 0, 1)
            loss = criterion(output, label)  #focal
            val_loss += loss.detach().cpu().item()
            val_accuracy += torch.sum(pred == label)

            val_label += label.detach().cpu().numpy().tolist()
            val_pred += pred.detach().cpu().numpy().tolist()

    val_loss = val_loss / len(valloader)
    val_accuracy = val_accuracy.float() / len(valloader)
    tp, fn, fp, tn = confusion_matrix(val_label, val_pred).ravel()
    tnr, fpr, fnr, tpr = tn/(fp+tn), fp/(fp+tn), fn/(tp+fn), tp/(tp+fn)
            
    return val_loss, val_accuracy, tnr, tpr

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
    
    # training parameter
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default='AdamW'
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
    )

    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    return args

if __name__ == '__main__' :
    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device = ", device)

    for k in range(1, 6):
    
        ##### Setting Seed #####
        seed = 11
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False

        model = convnext_tiny(len_of_clinical_features=150)
        model.to(device)

        weights = get_label_weights(args, k)
        criterion = FocalLoss(gamma=2, weights=weights).to(device)
        optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=20,
                                                cycle_mult=1.0,
                                                max_lr=args.lr,
                                                min_lr=0.000001,
                                                warmup_steps=10,
                                                gamma=1.0)

        # Data
        trainset = Lung_Dataset(
            images_path=args.input_dir, 
            clinical_data_path=args.clinical_data_path,
            train_val_test_list_path=args.train_val_test_path, 
            mode="train",
            k=k,
        )
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        valset = Lung_Dataset(
            images_path=args.input_dir,
            clinical_data_path=args.clinical_data_path,
            train_val_test_list_path=args.train_val_test_path,
            mode="val",
            k=k,
        ) 
        valloader = DataLoader(valset, batch_size=1, shuffle=False)
        
        train(args, k, device, model, optimizer, criterion, scheduler, trainloader, valloader, seed)