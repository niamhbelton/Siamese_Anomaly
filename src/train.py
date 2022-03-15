import torch
from datasets.main import load_dataset
from model import LeNet_Avg, LeNet_Max, LeNet_Tan, LeNet_Leaky, LeNet_Norm, LeNet_Drop, cifar_lenet
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

        a = (self.margin - euclidean_distance).unsqueeze(0)
        b=(torch.zeros(euclidean_distance.shape[0])).cuda().unsqueeze(0)
        c=torch.cat((a,b), dim=0)
        d=torch.max(c, dim=0).values

       # a = torch.Tensor([ torch.zeros(euclidean_distance.shape[0]), self.margin - euclidean_distance])
      #  e=torch.max(torch.Tensor([ torch.zeros(1,euclidean_distance.shape[0]), self.margin - euclidean_distance]))
      #  loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(d, 2) * 0.5)
        return loss_contrastive.sum()

def train(model, batch_size, train_dataset, val_dataset, epochs, criterion, model_name, indexes, data_path, normal_class, dataset_name):
    device='cuda'
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.000015, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=2, factor=.1, threshold=1e-4, verbose=True)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')
    best_val_auc = 0
    best_epoch = -1
    early_stop_iter = 0
    max_iter = 500
    stop_training =False
    ind = list(range(0, len(indexes)))

    train_losses = []
    val_losses = []
    aucs = []

    weight_totals = []

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)
        np.random.shuffle(ind)
        for i, index in enumerate(ind):


            seed = (epoch+1) * (index+1)
            img1, img2, labels = train_dataset.__getitem__(index, seed)
            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            output1 = model.forward(img1.float())
            output2 = model.forward(img2.float())

            if batch_size > 0:
              if (i ==0) | ((i-1) % batch_size == 0):
                outputs1 = output1
                outputs2 = output2
              else:
                outputs1 = torch.cat((outputs1, output1))
                outputs2 = torch.cat((outputs2, output2))

              if ((i % batch_size == 0) & (i !=0 ) )|( i == (len(ind)-1)):
                loss = criterion(outputs1,outputs2,labels)
                loss_sum+= loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:
                loss = criterion(output1,output2,labels)
                loss_sum+= loss.item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        total = 0
        for p in model.parameters():
            dims = p.size()
            n = p.cpu().data.numpy()
            total += np.sum(np.abs(n))
            weight_totals.append(total)

        print('weight total {}'.format(total))


        output_name = model_name + '_output_epoch_' + str(epoch+1)
        task = 'validate'
        val_auc, val_loss = evaluate(train_dataset, val_dataset, model, task, dataset_name, normal_class, output_name, indexes, data_path, criterion)

        aucs.append(val_auc)
        val_losses.append(val_loss)
        train_losses.append((loss_sum / len(indexes)))

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))
        print("Validation loss: {}".format(val_losses[-1]))
        print('AUC is {}'.format(aucs[-1]))

        scheduler.step(val_loss)
        if val_auc > best_val_auc:
          best_val_auc = val_auc
          best_epoch = epoch+1
          early_stop_iter = 0
          model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3))
          for f in os.listdir('./outputs/models/'):
            if model_name in f :
                os.remove(f'./outputs/models/{f}')
          torch.save(model.state_dict(), './outputs/models/' + model_name_temp)
        else:
          early_stop_iter += 1
          if early_stop_iter == max_iter:
            stop_training = True

        if stop_training:
          break

    if not os.path.exists('graph_data'):
        os.makedirs('graph_data')
    pd.concat([pd.DataFrame(weight_totals), pd.DataFrame(train_losses),pd.DataFrame(val_losses), pd.DataFrame(aucs)], axis =1).to_csv('./graph_data/' + model_name + '_epoch_' + str(epoch+1))

    print("Finished Training")
    print("Best validation AUC was {} on epoch {}".format(best_val_auc, best_epoch))




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
    parser.add_argument('--model_type', choices = ['LeNet_Avg', 'LeNet_Max', 'LeNet_Tan', 'LeNet_Leaky', 'LeNet_Norm', 'LeNet_Drop', 'cifar_lenet', 'MNIST_LeNet', 'LeNet5'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default = 0)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--seed', type=int, default = 100)
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
    batch_size = args.batch_size
    normal_class = args.normal_class
    N = args.num_ref
    seed = args.seed
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    indexes = args.index
    task = 'train'

    if indexes != []:
        indexes = [int(item) for item in indexes.split(', ')]
    else:
        indexes = create_reference(dataset_name, normal_class, task,  data_path, download_data, N, seed)


    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    val_dataset = load_dataset(dataset_name, indexes, normal_class, 'validate', data_path, download_data=False)

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
#    elif model_type == 'MNIST_LeNet':
#        model = MNIST_LeNet()
#    elif model_type == 'LeNet5':
#        model = LeNet5()
#    elif model_type == 'Net_Max':
#        model = Net_Max()
#    else:
#        model = Net_simp()



    model_name = model_name + '_normal_class_' + str(normal_class) + '_seed_' + str(seed)
    criterion = ContrastiveLoss()
    train(model, batch_size, train_dataset, val_dataset, epochs, criterion, model_name, indexes, data_path, normal_class, dataset_name)
