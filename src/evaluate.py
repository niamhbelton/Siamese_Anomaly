import torch.nn.functional as F
import torch
from model import LeNet_Avg, LeNet_Max, LeNet_Tan, LeNet_Leaky, LeNet_Norm, LeNet_Drop, cifar_lenet
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve
from sklearn import metrics
from datasets.main import load_dataset
import random


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
    #    print(label)
    #    if epoch > 100:
     #      print('ed is {}'.format(euclidean_distance))
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        return loss_contrastive



def evaluate(ref_dataset, model, task, dataset_name, normal_class, output_name, indexes, data_path, criterion):

    model.cuda()
    model.eval()


    #ref_dataset = Dataset(data_path, 'sagittal')
    #loader = torch.utils.data.DataLoader(ref_dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    #ref_dataset = load_dataset(dataset_name, indexes, normal_class, 'train', data_path, download_data=True)
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
    comp=[]
    for i in ind:

        img1, img2, label = ref_dataset.__getitem__(i)
        if label == 0:
          comp.append(i)
          d['compare{}'.format(i)] = img1 #ref_dataset.__getitem__(i)
          ref_images['images{}'.format(i)] = model.forward( d['compare{}'.format(i)].cuda().float())
          outs['outputs{}'.format(i)] =[]

    means = []
    lst=[]
    labels=[]
    #loop through images in the dataloader

    loss_sum =0
    for i, data in enumerate(loader):

        image = data[0][0]
        labels.append(data[2].item())
    #    lst.append(img[2].item())
        sum =0
        out = model.forward(image.cuda().float())
        for j in comp:
            euclidean_distance = F.pairwise_distance(out, ref_images['images{}'.format(j)])
            outs['outputs{}'.format(j)].append(euclidean_distance.detach().cpu().numpy()[0])
            sum += euclidean_distance.detach().cpu().numpy()[0]
            loss_sum += criterion(out, ref_images['images{}'.format(j)], data[2].item())

        means.append(sum / len(ind))
        del image
        del out



    df = pd.concat([pd.DataFrame(labels), pd.DataFrame(means)], axis =1)
    cols = ['label','mean']
    for i in comp:
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))

    df.columns=cols
    df = df.sort_values(by='mean', ascending = False).reset_index(drop=True)
    df.to_csv('./outputs/' +output_name)
    print('Writing output to {}'.format(('./outputs/' +output_name)))
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),softmax(np.array(df['mean'])))
    auc = metrics.auc(fpr, tpr)
    print('AUC is {}'.format(auc))

    return auc, loss_sum


def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)



def create_reference(dataset_name, normal_class, task, data_path, download_data, N, seed, few_shot):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0]
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N)
    final_indexes = ind[samp]
    if few_shot == True:
      for i in range(0,10):
          if i != normal_class:
            anom_ind = np.where(np.array(train_dataset.targets)==i)[0]
            random.seed(seed)
            s = random.sample(range(0, len(anom_ind)), 10)
            final_indexes = np.append(final_indexes, anom_ind[s])
    return final_indexes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, default = 'test', choices = ['train', 'test', 'validate'])
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-o', '--output_name', type=str, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--few_shot', default = False)
    parser.add_argument('--seed', type=int, default = 100)
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
    N = args.num_ref
    seed=args.seed
    few_shot = args.few_shot
    indexes = args.index

    #meta = pd.read_csv('metadata.csv')
    #if args.index != []:
    if indexes != []:
       indexes = [int(item) for item in args.index.split(', ')]
    else:
        download_data = False
        indexes = create_reference(dataset, normal_class, 'train', data_path, download_data, N, seed, few_shot)
        #else:
      #    indexes = list(meta.loc[meta['ref_set']==1, 'id'])

    #test_ind = list(meta.loc[meta['test']==1, 'id'])
    model = torch.load('./outputs/' + model_name)
    criterion = ContrastiveLoss()
    ref_dataset = load_dataset(dataset, indexes, normal_class, 'train', data_path, download_data=True)
    evaluate(ref_dataset, model, task, dataset, normal_class, output_name, indexes, data_path , criterion)
