import os
import random 
import shutil 
import argparse
import pandas as pd

# from os import shutil

def split_file(ori_path, output_dir, ratio):
    files = os.listdir(ori_path)
    fileIDs = [filename.split(".")[0] for filename in files]
    subfiles = os.listdir(output_dir)
    print(f"len_files: {len(files)}, len_subfiles: {len(subfiles)}")
    sample_num = int(len(files) * ratio)
    samples = random.sample(files, sample_num)
    for sample in samples: 
        # print('sample:', sample)
        shutil.copyfile(os.path.join(ori_path, sample), os.path.join(output_dir, sample))
    return 

def make_csv(filepath):
    files = os.listdir('./subset')
    fileIDs = [filename.split(".")[0] for filename in files]
    df = pd.DataFrame(fileIDs)
    df.to_csv('./subtrain.csv', index=False)
    # train = pd.read_csv('./train.csv', header=0)
    # text = []
    # labels = []
    # for ID in fileIDs:
    #     data = train.loc[train['id'] == ID]
    #     temp_text = data.loc[:, 'discourse_text']
    #     temp_labels = data.loc[:, 'discourse_type']
    #     for i in range(len(data)):
    #         text.append(temp_text.iloc[i])
    #         labels.append(temp_labels.iloc[i])
    # print(len(text))
    # print(len(train))
    # print(text.iloc[0])
    # print(labels)
    # print(temp)
    # for ID in fileIDs 
    # train.loc[train['id']]


    # print(train.loc[3]['id'])
    # count = 0
    # index = 0
    # n = len(train)
    # for i in range(n):
    #     ID  = train.loc[count]['id']
    #     if ID in fileIDs:
    #         count += 1
    #     else:
    #         train.drop(train.loc[count], inplace=True)
    # for ID in fileIDs:
    #     print(ID)
    #     train.drop(index=ID, level=1)
    # print(len(train))
    # train.to_csv('./sub_train.csv')

if __name__ == '__main__':
    parser = argparse.Argument()
    parser.add_argument('-i', '--input_dir', type=str, default='./train')
    parser.add_argument('-o', '--output_dir', type=str, default='./subset')
    args = parser.parse_args()
    ori_path = './train'
    output_dir = './val'
    ratio = 0.1 # this is the percentage of validation data in the whole data set
    split_file(ori_path, output_dir, ratio)
    make_csv(output_dir)
