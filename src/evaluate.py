import torch.nn.functional as F
import torch
from model import Net
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve
from sklearn import metrics
from datasets.main import load_dataset


def evaluate(model, task, dataset_name, normal_class, output_name, indexes, data_path):

    model.cuda()
    model.eval()


    #ref_dataset = Dataset(data_path, 'sagittal')
    #loader = torch.utils.data.DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    ref_dataset = load_dataset(dataset_name, indexes, normal_class, 'train', data_path, download_data=False)
   # if task == 'test':
    #    test = load_dataset(dataset_name, indexes, normal_class, 1, data_path, download_data=False)
  #  else: #evaluate on validation data
    test = load_dataset(dataset_name, indexes, normal_class, task, data_path, download_data=False)
    loader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=False, num_workers=1, drop_last=False)
    d={} #a dictionary of the reference images
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, len(indexes)))
    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    for i in ind:
        d['compare{}'.format(i)],_,_ = ref_dataset.__getitem__(i)
        ref_images['images{}'.format(i)] = model.forward( d['compare{}'.format(i)].cuda().float())
        outs['outputs{}'.format(i)] =[]

    means = []
    lst=[]
    labels=[]
    #loop through images in the dataloader
    for i, data in enumerate(loader):

        image = data[0][0]
        labels.append(data[2].item())
    #    lst.append(img[2].item())
        sum =0
        out = model.forward(image.cuda().float())
        for j in range(0, len(indexes)):
            euclidean_distance = F.pairwise_distance(out, ref_images['images{}'.format(j)])
            outs['outputs{}'.format(j)].append(euclidean_distance.detach().cpu().numpy()[0])
            sum += euclidean_distance.detach().cpu().numpy()[0]
        means.append(sum / len(indexes))
        del image
        del out

    df = pd.concat([pd.DataFrame(labels), pd.DataFrame(means)], axis =1)
    cols = ['label','mean']
    for i in range(0, len(indexes)):
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))

    df.columns=cols
    df = df.sort_values(by='mean', ascending = False).reset_index(drop=True)
    df.to_csv('./outputs/' +output_name)
    print('Writing output to {}'.format(('./outputs/' +output_name)))
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),softmax(np.array(df['mean'])))
    auc = metrics.auc(fpr, tpr)
    print('AUC is {}'.format(auc))


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, default = 'test', choices = ['train', 'test', 'validate'])
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-o', '--output_name', type=str, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_arguments()
    model_name = args.model_name
    dataset = args.dataset
    task = args.task
    normal_class = args.normal_class
    output_name = args.output_name
    data_path = args.data_path

    #meta = pd.read_csv('metadata.csv')
    #if args.index != []:
    indexes = [int(item) for item in args.index.split(', ')]
    #else:
    #    indexes = list(meta.loc[meta['ref_set']==1, 'id'])

    #test_ind = list(meta.loc[meta['test']==1, 'id'])
    model = torch.load('./outputs/' + model_name)
    evaluate(model, task, dataset, normal_class, output_name, indexes, data_path )
