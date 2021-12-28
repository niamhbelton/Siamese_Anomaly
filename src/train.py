import torch
from datasets.main import load_dataset
from model import Net, Net_simp, cifar_lenet, MNIST_LeNet
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate
import random


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        return loss_contrastive

def train(model, train_dataset, epochs, criterion, model_name, indexes, data_path, normal_class, dataset_name):
    device='cuda'
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    ind = list(range(0, len(indexes)))
    for epoch in range(epochs):
        model.train()
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)
        np.random.shuffle(ind)
        for index in ind:
            img1, img2, labels = train_dataset.__getitem__(index)
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            output1 = model.forward(img1.float())
            output2 = model.forward(img2.float())
            loss = criterion(output1,output2,labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        output_name = 'output_epoch_' + str(epoch)
        task = 'validate'
        evaluate(model, task, dataset_name, normal_class, output_name, indexes, data_path)



    torch.save(model, './outputs/' + model_name)
    print("Finished Training")




def create_reference(dataset_name, normal_class, task, data_path, download_data, N):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0]
    samp = random.sample(range(0, len(ind)), N)

    return ind[samp]



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['Net', 'Net_simp', 'cifar_lenet', 'MNIST_LeNet'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model_name = args.model_name
    model_type = args.model_type
    dataset_name = args.dataset
    normal_class = args.normal_class
    N = args.num_ref
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    indexes = args.index
    task = 'train'

    if indexes != []:
        indexes = [int(item) for item in indexes.split(', ')]
    else:
        indexes = create_reference(dataset_name, normal_class, task,  data_path, download_data, N)


    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)

    if model_type == 'Net':
        model = Net()
    elif model_type == 'cifar_lenet':
        model = cifar_lenet()
    elif model_type == 'MNIST_LeNet':
        model = MNIST_LeNet()
    else:
        model = Net_simp()


    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.1)
    criterion = ContrastiveLoss()
    train(model, train_dataset, epochs, criterion, model_name, indexes, data_path, normal_class, dataset_name)
