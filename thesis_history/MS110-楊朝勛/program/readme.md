# Preprocess
## Dual Energy CT

1. 將 dual energy CT 從 dicom 轉 numpy 格式

ˋˋˋshell
python3 Dicom2Numpy.py

特別注意:
最原始的資料集 (dicom format) 需要去實驗室 CAD SERVER 的 /Lung CT/NTU/GSI 下載，
過程中可能還需要一些些調整 (主要是改路徑或改名稱)，
方能用這個程式將 dicom 轉成 numpy 格式
ˋˋˋ

2. WindowSetting (-1000 ~ 400)

ˋˋˋshell
python3 WindowSetting.py \
    --input_path ../../../data/C+_140/raw_data/ \
    --output_path ../../../data/C+_140/clip/

parameter introduction:

--input_path path/to/input/image/directory (numpy format)
--output_path path/to/output/image/directory (numpy format)
ˋˋˋ

3. 依照 VOI 切割 dual energy CT

ˋˋˋshell
python3 VoiCropping.py \
    --label_path ../../Lung GSI patient list_20220818_查資料.xlsx \
    --input_path ../../../data/C+_140/clip/ \
    --output_path ../../../data/C+_140/voi_128_224_224/

parameter introduction:
--label_path path/to/VOI/information/file
--input_path path/to/input/image/directory (numpy format)
--output_path path/to/output/image/directory (numpy format)
ˋˋˋ

## Clinical Data

1. 將臨床資訊轉成值只有 0 和 1 的資料 (one hot encoding)

ˋˋˋshell
python3 OneHotEncoding.py \
    --input_path ../../Lung GSI patient list_20220818_查資料.xlsx \
    --output_path ../../transformed_clinical_data.xlsx

parameter introduction:
--input_path path/to/clinical/information/file
--output_path path/to/output/file (excel)
ˋˋˋ

# Classification

1. 訓練模型

ˋˋˋshell
python3 main.py \
    --clinical_data_path: ../transformed_clinical_data.xlsx \
    --input_dir: ../../data/C+_140/voi_128_224_224 \
    --train_val_test_path: ../train_val_test_split.xlsx \
    --model_dir: ./model \
    --batch_size 16 \
    --epochs 100 \
    --weight_decay: 1e-1 \
    --lr: 0.001 \
    --optimizer AdamW \
    --patience: 5

parameter introduction:

--clinical_data_path: 臨床資料的位置，已經經過 one hot encoding
--input_dir: dual energy CT 前處理完的資料夾路徑
--train_val_test_path: train, val, test data 紀錄表 (excel)
--model_dir: 欲儲存模型的位置
--batch_size
--epochs
--weight_decay: optimizer 的參數
--lr: 訓練中最高的 learning rate
--optimizer
--patience: Validation loss 連續 patience 次沒下降，則中斷訓練
ˋˋˋ

2. 測試模型

ˋˋˋshell
python3 test.py \
    --clinical_data_path ../transformed_clinical_data.xlsx \
    --input_dir ../../data/C+_140/voi_128_224_224 \
    --train_val_test_path ../train_val_test_split.xlsx \
    --model_dir ./model \
    --output_dir ./3year_result/ \
    --nfold 5

parameter introduction:

--clinical_data_path: 臨床資料的位置，已經經過 one hot encoding
--input_dir: dual energy CT 前處理完的資料夾路徑
--train_val_test_path: train, val, test data 紀錄表 (excel)
--model_dir: 儲存模型的位置
--output_dir: 欲儲存準確率的資料夾
--nfold 5: number of fold (cross validation)
ˋˋˋ

3. 其他程式碼

ˋˋˋshell
ConvNeXt.py: ConvNeXt model
dataset.py: 將 dual energy CT 與 clinical data 轉換成 pytorch 的資料集
other_blocks.py: attention blocks
torch_focal_loss.py: focal loss
ˋˋˋ

For quick demo
```shell
cd under/Classification/dir
python3 main.py
python3 test.py
```

For 完整流程
```shell
cd under/Prepocess/Dual Energy CT/dir
python3 Dicom2Numpy.py (資料在 CAD SERVER 的其他路徑)
python3 WindowSetting.py
python3 VoiCropping.py

cd under/Prepocess/Clinical Data/dir
python3 OneHotEncoding.py

cd under/Classification/dir
python3 main.py
python3 test.py
```

