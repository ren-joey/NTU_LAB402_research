from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple
from argparse import ArgumentParser, Namespace
from collections import Counter
import logging
import random

logging.basicConfig(level=logging.INFO)

TRAIN = "train"
VAL = "validation"
TEST = "test"
SPLITS = [TRAIN, VAL, TEST]

def split_data(labels: Counter, cv: int, output_path: Path) -> None:
    ndata = {0 : (21, (6, 6, 6, 6, 7), (7, 7, 7, 7, 6)), 1 : (128, (34, 34, 34, 34, 32), (40, 40, 40, 40, 42))}
    nSum = {0 : 34, 1 : 202}
    output: Dict[str, List] = {SPLIT+str(fold): [] for SPLIT in SPLITS for fold in range(1, cv+1)}

    for label in labels:
        train_ids = labels[label]
        start = 0
        end = 0
        new_fold_start = 0

        for fold in range(1, cv+1):

            start = new_fold_start
            end = start + ndata[label][2][fold-1]
            if end >= nSum[label]:
                output[TEST+str(fold)] += [id for id in train_ids[start : nSum[label]]]
                end = end % nSum[label]
                output[TEST+str(fold)] += [id for id in train_ids[0 : end]]
            else:
                output[TEST+str(fold)] += [id for id in train_ids[start : end]]

            start = end
            new_fold_start = end
            end = start + ndata[label][1][fold-1]
            if end >= nSum[label]:
                output[VAL+str(fold)] += [id for id in train_ids[start : nSum[label]]]
                end = end % nSum[label]
                output[VAL+str(fold)] += [id for id in train_ids[0 : end]]
            else:
                output[VAL+str(fold)] += [id for id in train_ids[start : end]]

            start = end
            end = start + ndata[label][0]
            if end >= nSum[label]:
                tmp = [id for id in train_ids[start : nSum[label]]]
                end = end % nSum[label]
                tmp += [id for id in train_ids[0 : end]]
            else:
                tmp = [id for id in train_ids[start : end]]
            if label == 0:
                output[TRAIN+str(fold)] += 6 * tmp
            else:
                output[TRAIN+str(fold)] += tmp

        print(f'label = {label}')
        for key, out in output.items():
            print(f'{key} num = {len(out)}')
            
    df = pd.DataFrame.from_dict(output, orient='index').T
    df.to_excel(output_path, index=False) 

    logging.info(f"Split file saved at {output_path.resolve()}")
    
def main(args):
    df = pd.read_excel(args.clinical_data_path, usecols="B, I")
    # change header
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    clinical_data : Dict[str, int] = df.set_index('Patient ID').to_dict()['是否存活']
    # print(clinical_data)

    labels = Counter()
    for id in clinical_data:
        labels.setdefault(clinical_data[id], list()) # e.g. {1: [], 0: []}
        labels[clinical_data[id]].append(id)

    print(labels)
    split_data(labels, args.cv, args.output_path)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--clinical_data_path",
        type=Path,
        help="Path to the clinical data.",
        default="../Lung GSI patient list_20220818_查資料.xlsx",
    )

    # output csv
    parser.add_argument(
        "--output_path",
        type=Path,
        help="Directory to save the data",
        default="../train_val_test_split.xlsx",
    )

    parser.add_argument(
        "--cv",
        type=int,
        help="number of cross validation",
        default=5
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)