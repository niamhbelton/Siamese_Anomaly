import torch.nn.functional as F
import torch
from model import LeNet_Avg, LeNet_Max, LeNet_Tan, LeNet_Leaky, LeNet_Norm, LeNet_Drop, CIFAR_VGG3, MNIST_VGG3
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve
from sklearn import metrics
from datasets.main import load_dataset
import random





def evaluate(feat1, base_ind, ref_dataset, val_dataset, model, task, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha):

    model.eval()


    #create loader for dataset that is testing
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, len(indexes)))
    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.


    cols=[]

    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    for i in ind:
      img1, _, _, _ = ref_dataset.__getitem__(i)
      if i == base_ind:
        ref_images['images{}'.format(i)] = feat1
      else:
        ref_images['images{}'.format(i)] = model.forward( img1.cuda().float())

    #  ref_images['images{}'.format(i)] = model.forward( img1.cuda().float())
      outs['outputs{}'.format(i)] =[]

      string = 'col_' + str(i)
      cols.append(string)


    means = []
    minimum_dists=[]
    lst=[]
    labels=[]



    #loop through images in the dataloader
    loss_sum =0
    for i, data in enumerate(loader):

        if i % 1000 == 0:
          print(i)

        image = data[0][0]
        label = data[2].item()

        labels.append(label)
        total =0
        mini=torch.Tensor([1e50])
        maxi = torch.Tensor([0])

        out = model.forward(image.cuda().float()) #get feature vector for test image


        for j in range(0, len(indexes)):
            euclidean_distance = (F.pairwise_distance(out, ref_images['images{}'.format(j)]) / torch.sqrt(torch.Tensor([out.size()[1]])).cuda() ) + (alpha*(F.pairwise_distance(out, feat1) /torch.sqrt(torch.Tensor([out.size()[1]])).cuda() ))
            outs['outputs{}'.format(j)].append(euclidean_distance.item())
            total += euclidean_distance.item()
            if euclidean_distance.detach().item() < mini:
              mini = euclidean_distance.item()

            if euclidean_distance.item() > maxi:
              maxi = euclidean_distance.item()

            loss_sum += criterion(out,[ref_images['images{}'.format(j)]], feat1,label, alpha,weight=False).item()

        minimum_dists.append(mini)
        means.append(total/len(indexes))


        del image
        del out
        del euclidean_distance
        del total
        torch.cuda.empty_cache()


    cols = ['label','minimum_dists', 'means']
    df = pd.concat([pd.DataFrame(labels, columns = ['label']), pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)


    print('the mean of anoms is {}'.format(np.mean(df['means'].loc[df['label'] == 1])))
    print('the mean of normals is {}'.format(np.mean(df['means'].loc[df['label'] == 0])))
    for i in range(0, len(indexes)):
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))

    df.columns=cols
    df = df.sort_values(by='minimum_dists', ascending = False).reset_index(drop=True)
 #   for f in os.listdir('./outputs/ED/'):
  #    if model_name in f :
   #       os.remove(f'./outputs/ED/{f}')
    #df.to_csv('./outputs/ED/' +output_name)

    if task != 'train':
        fpr, tpr, thresholds = roc_curve(np.array(df['label']),softmax(np.array(df['minimum_dists'])))
        auc_min = metrics.auc(fpr, tpr)
        print('AUC based on minimum vector {}'.format(auc_min))
        fpr, tpr, thresholds = roc_curve(np.array(df['label']),softmax(np.array(df['means'])))
        auc = metrics.auc(fpr, tpr)

    feat_vecs = pd.DataFrame(ref_images['images1'].detach().cpu().numpy())
    for j in range(1, len(indexes)):
        feat_vecs = pd.concat([feat_vecs, pd.DataFrame(ref_images['images{}'.format(j)].detach().cpu().numpy())], axis =0)

    avg_loss = (loss_sum / len(indexes) )/ val_dataset.__len__()
    return auc, avg_loss, auc_min, df, feat_vecs

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)



def create_reference(dataset_name, normal_class, task, data_path, download_data, N, seed):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0]
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N)
    final_indexes = ind[samp]
    return final_indexes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--task', type=str, required=True, default = 'test', choices = ['train', 'test', 'validate'])
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('--model_type', choices = ['LeNet_Avg', 'LeNet_Max', 'LeNet_Tan', 'LeNet_Leaky', 'LeNet_Norm', 'LeNet_Drop', 'cifar_lenet', 'MNIST_LeNet', 'LeNet5'], required=True)
    parser.add_argument('-o', '--output_name', type=str, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)

    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args



if __name__ == '__main__':

    from train import ContrastiveLoss
    from train import init_feat_vec

    args = parse_arguments()
    model_name = args.model_name
    dataset = args.dataset
    task = args.task
    normal_class = args.normal_class
    model_type = args.model_type
    output_name = args.output_name
    data_path = args.data_path
    N = args.num_ref
    seed=args.seed
    indexes = args.index
    alpha = args.alpha


    if indexes != []:
       indexes = [int(item) for item in args.index.split(', ')]
    else:
        download_data = False
        indexes = create_reference(dataset, normal_class, 'train', data_path, download_data, N, seed)

    if model_type == 'LeNet_Avg':
        model = LeNet_Avg()
    elif model_type == 'LeNet_Max':
        model = LeNet_Max()
    elif model_type == 'LeNet_Tan':
        model = LeNet_Tan()
    elif model_type == 'LeNet_Leaky':
        model = LeNet_Leaky()
    elif model_type == 'LeNet_Norm':
        model = LeNet_Norm()
    elif model_type == 'LeNet_Drop':
        model = LeNet_Drop()
#    if model_type == 'Net':
#        model = Net()
    elif model_type == 'cifar_lenet':
        model = cifar_lenet()

    model.load_state_dict(torch.load('./outputs/models/' + model_name))
    model.cuda()

    criterion = ContrastiveLoss()
    ref_dataset = load_dataset(dataset, indexes, normal_class, 'train', data_path, download_data=True)
    val_dataset = load_dataset(dataset, indexes, normal_class, task, data_path, download_data=False)
    ind=list(range(0,len(indexes)))
    np.random.seed(50)
    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]
    feat1 = init_feat_vec(model,base_ind, ref_dataset )
    auc, loss, vec_sum, vec_mean, feature_vectors, feature_vectors2, test_vectors = evaluate(feat1, base_ind,ref_dataset, val_dataset, model, task, dataset, normal_class, output_name, indexes, data_path , criterion, alpha)

    print('AUC is {}'.format(auc))
