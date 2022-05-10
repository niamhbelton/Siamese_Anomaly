import torch
from datasets.main import load_dataset
from model import CIFAR_VGG3, MNIST_VGG3
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate
import random
import time

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, v=0.0,margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.v = v

    def forward(self, output1, vectors, feat1, label, alpha):
        euclidean_distance = torch.FloatTensor([0]).cuda()

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)) / torch.sqrt(torch.Tensor([output1.size()[1]])).cuda()

        euclidean_distance += alpha*((F.pairwise_distance(output1, feat1)) /torch.sqrt(torch.Tensor([output1.size()[1]])).cuda() )

        #calculate the margin
        marg = (len(vectors) + alpha) * self.margin
        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss
        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)
        return loss_contrastive

def train(model, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp, k):
    device='cuda'
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_losses = []

    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    if freeze == True:
      np.random.seed(epochs)
      rand_freeze = np.random.randint(len(indexes) )
      base_ind = ind[rand_freeze]
      feat1 = init_feat_vec(model,base_ind , train_dataset)

    patience = 0
    max_patience = 2
    start_time = time.time()

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

              loss = criterion(output1,output2,feat1,labels,alpha)

            else:
              max_eds = [0] * k
              max_inds = [-1] * k
              max_ind =-1
              vectors=[]
              for j in range(0, len(ind)):
                if (ind[j] != base_ind) & (index != ind[j]):
                  output2=model(train_dataset.__getitem__(ind[j], seed, base_ind)[0].to(device).float())
                  vectors.append(output2)
                  euclidean_distance = F.pairwise_distance(output1, output2)
                  for b, vec in enumerate(max_eds):
                      if euclidean_distance > vec:
                        max_eds.insert(b, euclidean_distance)
                        max_inds.insert(b, len(vectors)-1)
                        if len(max_eds) > k:
                          max_eds.pop()
                          max_inds.pop()
                        break

              loss = criterion(output1,[vectors[x] for x in max_inds],feat1,labels,alpha)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        train_losses.append((loss_sum / len(indexes)))

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))

        if epoch > 1:
          decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)
        #  print((((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100))
         # print((((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100))
          #print(decrease)

          if decrease <= 1.0:
            patience += 1


          if (patience==max_patience):
            print("--- %s seconds ---" % (time.time() - start_time))
            training_time = time.time() - start_time
            task = 'validation'
            output_name = model_name + '_output_epoch_' + str(epoch+1)
            val_auc, val_loss, val_auc_min, df, ref_vecs = evaluate(feat1, base_ind, train_dataset, val_dataset, model, task, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha)

            model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
            for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                  os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
            torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)
            for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                  os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
            df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

            for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
              if (model_name in f) :
                os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
            ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)

            break



    print("Finished Training")
    print("AUC was {} on epoch {}".format(val_auc, epoch))


    return val_auc, epoch, val_auc_min, training_time



def init_feat_vec(model,base_ind, train_dataset ):

        model.eval()
        feat1,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          feat1 = model(feat1.cuda().float()).cuda()

        return feat1



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data) #get all training data
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.ceil(N*contamination)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0] #get indexes of non-normal class
      samp = random.sample(range(0, len(con)), int(numb))
      samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
      final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','MNIST_VGG3'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 20)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--freeze', default = True)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
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
    vector_size = args.vector_size
    weight_decay = args.weight_decay
    smart_samp = args.smart_samp
    k = args.k
    weight_init_seed = args.weight_init_seed
    v = args.v
    task = 'train'

    #if indexes for reference set aren't provided, create the reference set.
    if indexes != []:
        indexes = [int(item) for item in indexes.split(', ')]
    else:
        indexes = create_reference(contamination, dataset_name, normal_class, task,  data_path, download_data, N, seed)

    #create train and test set
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    val_dataset = load_dataset(dataset_name, indexes, normal_class, 'validate', data_path, download_data=False)

    #set the seed
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)

    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')

    string = './outputs/models/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')

    string = './outputs/ED/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/ref_vec'):
        os.makedirs('outputs/ref_vec')

    string = './outputs/ref_vec/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    #Initialise the model
    if model_type == 'CIFAR_VGG3':
        model = CIFAR_VGG3(vector_size)
    elif model_type == 'MNIST_VGG3':
        model = MNIST_VGG3(vector_size)


    model_name = model_name + '_normal_class_' + str(normal_class) + '_seed_' + str(seed)
    criterion = ContrastiveLoss(v)
    auc, epoch, auc_min, training_time = train(model,lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, freeze, smart_samp,k)

    #write out all details of model training
    cols = ['normal_class', 'ref_seed', 'weight_seed', 'alpha', 'lr', 'weight_decay', 'vector_size', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time']
    params = [normal_class, seed, weight_init_seed, alpha, lr, weight_decay, vector_size, smart_samp, k, v, contamination, auc, epoch, auc_min, training_time]
    string = './outputs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name)
