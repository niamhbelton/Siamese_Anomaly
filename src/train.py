import torch
from datasets.main import load_dataset
from model import LeNet_Avg, LeNet_Max, LeNet_Tan, LeNet_Leaky, LeNet_Norm, LeNet_Drop, cifar_lenet, cifar_lenet_red
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

    def forward(self, output1, vectors, feat1, label, alpha, weight=False, task=False):
        euclidean_distance = torch.FloatTensor([0]).cuda()
        if weight == True:
          eds=[]
          for i in vectors:
            eds.append(F.pairwise_distance(i, feat1))

          for i,dist in enumerate(eds):
            euclidean_distance += ((1-(dist/sum(eds))) * F.pairwise_distance(output1, vectors[i]))

          euclidean_distance += (alpha*F.pairwise_distance(output1, feat1))
        else:
          for i in vectors:
            euclidean_distance += F.pairwise_distance(output1, i) #+ ((alpha*F.pairwise_distance(output1, feat1)))))
          euclidean_distance += alpha*F.pairwise_distance(output1, feat1)



        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), self.margin - euclidean_distance])), 2) * 0.5)
        return loss_contrastive

def train(model, lr, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp, k, weight):
    device='cuda'
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=2, factor=.1, threshold=1e-4, verbose=True)
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')
    if not os.path.exists('graph_data'):
        os.makedirs('graph_data')

    best_val_auc = 0
    best_epoch = -1
    best_val_auc_min = 0
    best_epoch_min = -1
    early_stop_iter = 0
    max_iter = 5
    stop_training =False
    ind = list(range(0, len(indexes)))

    train_losses = []
    val_losses = []
    aucs = []

    weight_totals = []
    weight_means = []


    np.random.seed(epochs)
    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]

    dictionary={}
    for i in range(0, len(ind)):
      dictionary[i] = []


    if freeze == True:
      feat1 = init_feat_vec(model,base_ind , train_dataset)

    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        print("Starting epoch " + str(epoch+1))
        np.random.seed(epoch)
        np.random.shuffle(ind)
        for i, index in enumerate(ind):

            seed = (epoch+1) * (index+1)
            img1, img2, labels, base = train_dataset.__getitem__(index, seed, base_ind)

            # Forward
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            if (freeze == True) & (index ==base_ind):
              output1 = feat1
            else:
              output1 = model.forward(img1.float())

            if smart_samp == 0:
              if (freeze == True) & (base == True):
                output2 = feat1
              else:
                output2 = model.forward(img2.float())

              loss = criterion(output1,output2,feat1,labels,alpha,True)

            else:
              max_eds = [0] * k
              max_inds = [-1] * k
              max_euclidean_distance =0
              max_ind =-1
              vectors=[]
              for j in range(0, len(ind)):
                if ind[j] == base_ind:
                  vectors.append(feat1)
                  euclidean_distance = F.pairwise_distance(output1, feat1)
                else:
                  output2=model(train_dataset.__getitem__(ind[j], seed, base_ind)[0].to(device).float())
                  vectors.append(output2)
                  euclidean_distance = F.pairwise_distance(output1, output2)
                if smart_samp == 1:
                  for b, vec in enumerate(max_eds):
                    if euclidean_distance > vec:
                      max_eds.insert(b, euclidean_distance)
                      max_inds.insert(b, j)
                      if len(max_eds) > k:
                        max_eds.pop()
                        max_inds.pop()
                      break


                else:
                  if (euclidean_distance > max_euclidean_distance) & (j not in dictionary[index]):
                    max_euclidean_distance = euclidean_distance
                    max_ind = j

              dictionary[index].append(max_ind)
          #    print(max_inds)
          #    print(vectors[[max_inds]])
              loss = criterion(output1,[vectors[x] for x in max_inds],feat1,labels,alpha,weight=weight,True)


        #    if i == 3:
        #      loss = criterion(output1,output2,feat1,labels,True)
        #    else:

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()



        output_name = model_name + '_output_epoch_' + str(epoch+1)
        task = 'validate'
        val_auc, val_loss, val_auc_min, vec_sum, vec_mean, feature_vectors, feature_vectors2, test_vectors = evaluate(feat1, base_ind, train_dataset, val_dataset, model, task, dataset_name, normal_class, output_name, indexes, data_path, criterion, alpha)

        aucs.append(val_auc)
        val_losses.append(val_loss)
        train_losses.append((loss_sum / len(indexes)))

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))
        print("Validation loss: {}".format(val_losses[-1]))
        print('AUC is {}'.format(aucs[-1]))


        scheduler.step(val_loss)
        best_model=False
        if val_auc > best_val_auc:
          best_val_auc = val_auc
          best_epoch = epoch+1
          early_stop_iter = 0
          best_model=True
          model_name_temp = model_name + '_mean_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3))
          for f in os.listdir('./outputs/models/'):
            if (model_name in f) & ('mean' in f) :
                os.remove(f'./outputs/models/{f}')
          torch.save(model.state_dict(), './outputs/models/' + model_name_temp)


        if val_auc_min > best_val_auc_min :
          best_val_auc_min = val_auc_min
          best_epoch_min = epoch+1

          early_stop_iter = 0
          best_model=True
          model_name_temp = model_name + '_minimum_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc_min, 3))
          for f in os.listdir('./outputs/models/'):
            if (model_name in f) & ('minimum' in f):
                os.remove(f'./outputs/models/{f}')
          torch.save(model.state_dict(), './outputs/models/' + model_name_temp)

        if best_model==False:
          early_stop_iter += 1
          if early_stop_iter == max_iter:
            stop_training = True



        if (i % 20 == 0) | (stop_training == True):

          for f in os.listdir('graph_data'):
            if model_name in f :
                os.remove(f'./graph_data/{f}')
          pd.concat([pd.DataFrame(weight_totals), pd.DataFrame(weight_means), pd.DataFrame(train_losses),pd.DataFrame(val_losses), pd.DataFrame(aucs)], axis =1).to_csv('./graph_data/' + model_name + '_epoch_' + str(epoch+1))
          pd.concat([pd.DataFrame(vec_sum, columns = ['sum_abs_vals']),pd.DataFrame(vec_mean, columns = ['mean_vals']),feature_vectors ], axis =1).to_csv('./graph_data/vectors_' + output_name)
          pd.concat([pd.DataFrame(vec_sum, columns = ['sum_abs_vals']),pd.DataFrame(vec_mean, columns = ['mean_vals']),feature_vectors2 ], axis =1).to_csv('./graph_data/vectors_orig_' + output_name)
          test_vectors.to_csv('./graph_data/test_vectors_' + output_name)


        if stop_training:
          break


    print("Finished Training")
    print("Best validation AUC was {} on epoch {}".format(best_val_auc, best_epoch))



def init_feat_vec(model,base_ind, train_dataset ):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""

        model.eval()
        feat1,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          feat1 = model(feat1.cuda().float()).cuda()

        return feat1



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0]
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N)
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.floor(N*contamination)
      print(numb)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0]
      samp = random.sample(range(0, len(con)), int(numb))
      final_indexes = np.array(list(final_indexes) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['LeNet_Avg', 'LeNet_Max', 'LeNet_Tan', 'LeNet_Leaky', 'LeNet_Norm', 'LeNet_Drop', 'cifar_lenet','cifar_lenet_red', 'MNIST_LeNet', 'LeNet5'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--freeze', default = True)
    parser.add_argument('--smart_samp', type = int, choices = [0,1,2], default = 0)
    parser.add_argument('--weight', type = bool, default = False)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
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
    seed = args.seed
    freeze = args.freeze
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    contamination = args.contamination
    indexes = args.index
    alpha = args.alpha
    lr = args.lr
    smart_samp = args.smart_samp
    k = args.k
    weight = args.weight
    weight_init_seed = args.weight_init_seed
    task = 'train'

    if indexes != []:
        indexes = [int(item) for item in indexes.split(', ')]
    else:
        indexes = create_reference(contamination, dataset_name, normal_class, task,  data_path, download_data, N, seed)


    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    val_dataset = load_dataset(dataset_name, indexes, normal_class, 'validate', data_path, download_data=False)


    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)
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
    elif model_type == 'cifar_lenet_red':
        model = cifar_lenet_red()
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
    train(model,lr, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp,k, weight)
