import pandas as pd
import numpy as np
import os
import argparse


def pre_csv(data_path, frac, output_path):
    np.random.seed(42)
    image_ids = os.listdir(data_path)
    data_size = len(image_ids)
    train_size = int(round(len(image_ids) * frac, 0))
    train_set = np.random.choice(image_ids,train_size,replace=False)
    ds_split = []
    for img_id in image_ids:
        if img_id in train_set:
            ds_split.append('train')
        else:
            ds_split.append('test')
    
    ds_dict = {'image_id':image_ids,
               'category':ds_split 
        }
    df = pd.DataFrame(ds_dict)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print('Number of train sample: {}'.format(len(train_set)))
    print('Number of test sample: {}'.format(data_size-train_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='the path of images')
    parser.add_argument('--output', type=str, required=True, help='output CSV path')
    parser.add_argument('--size', type=float, default=0.9, help='the size of your train set')
    args = parser.parse_args()
    pre_csv(args.dataset, args.size, args.output)
