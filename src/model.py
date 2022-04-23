import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


#lenet without last layers and average pooling
class LeNet_Avg(nn.Module):
    def __init__(self):
        super(LeNet_Avg, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
          #  nn.MaxPool2d(kernel_size=2),
            nn.AvgPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
           # nn.MaxPool2d(2),
           nn.AvgPool2d(kernel_size=2)
        )
        # fully connected layer, output 10 classes
  #      self.out = nn.Linear(32 * 7 * 7, 1)

    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x #output

#lenet without last layers and max pooling
class LeNet_Max(nn.Module):
    def __init__(self):
        super(LeNet_Max, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2, bias=False
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2, bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(1568, 1024, bias=False)

       # self.classifier = nn.Linear(1568, 512)

     #   self.classifier = nn.Linear(1568, 64)

      #  self.classifier = nn.Linear(1568, 2048)


    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x #output


class LeNet_Tan(nn.Module):
    def __init__(self):
        super(LeNet_Tan, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.Tanh(),
            nn.MaxPool2d(2)
        )


    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x #output


class LeNet_Leaky(nn.Module):
    def __init__(self):
        super(LeNet_Leaky, self).__init__()
        self.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)



    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return x #output


class LeNet_Norm(nn.Module):
    def __init__(self):
        super(LeNet_Norm, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x #output



class LeNet_Drop(nn.Module):
    def __init__(self):
        super(LeNet_Drop, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
        #    nn.BatchNorm2d(16, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            #nn.BatchNorm2d(32, eps=1e-04, affine=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )


    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return x #output



#more complicated version
class MNIST_LeNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.rep_dim = 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return x

#including a linear layer
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=2)
        self.classifier = nn.Linear(in_features=256, out_features=84)
        self.act = nn.Tanh()

    def forward(self, x):
        x = torch.unsqueeze(x, dim =0)
        x = torch.unsqueeze(x, dim =0)
        x= self.act(self.conv1(x))
        x = self.pool(x)
        x = self.act(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
     #   x= self.classifier(x)
        return x



class cifar_lenet(nn.Module):
  def __init__(self):
      super(cifar_lenet, self).__init__()

      self.conv1 =nn.Conv2d(
              in_channels=3,
              out_channels=32,
              kernel_size=3,
              stride=1,
              padding=0,bias=False
          )
      self.bn1 = nn.BatchNorm2d(32, eps=1e-04, affine=False)
      self.act = nn.LeakyReLU()
      self.conv2 =nn.Conv2d(
              in_channels=32,
              out_channels=32,
              kernel_size=3,
              stride=1,
              padding=0,bias=False
          )
      self.act2 = nn.LeakyReLU()
      self.pool=nn.MaxPool2d(kernel_size=2)

      self.conv3 =nn.Conv2d(
              in_channels=32,
              out_channels=64,
              kernel_size=3,
              stride=1,
              padding=0,bias=False
          )
      self.act3 = nn.LeakyReLU()
      self.conv4 =nn.Conv2d(
              in_channels=64,
              out_channels=64,
              kernel_size=3,
              stride=1,
              padding=0,bias=False
          )
      self.act4 = nn.LeakyReLU()
      self.pool2 = nn.MaxPool2d(kernel_size=2)


      self.classifier = nn.Linear(2880, 1024,bias=False)
      self.act5 = nn.LeakyReLU()
      self.drop = nn.Dropout(p=0.5)
      self.classifier2 = nn.Linear(1024, 512,bias=False)


      # self.classifier = nn.Linear(1568, 512)

    #   self.classifier = nn.Linear(1568, 64)

    #  self.classifier = nn.Linear(1568, 2048)


  def forward(self, x):
      x = torch.unsqueeze(x, dim =0)
      x = F.pad(x, (0, 0, 1, 1))
      x = self.conv1(x)
      x = self.bn1(x)
      x = self.act(x)
      x = F.pad(x, (0, 0, 1, 1))
      x = self.conv2(x)
      x = self.act2(x)
      x= self.pool(x)

      x = F.pad(x, (0, 0, 1, 1))
      x = self.conv3(x)
      x = self.act3(x)
      x = F.pad(x, (0, 0, 1, 1))
      x = self.conv4(x)
      x = self.act4(x)
      x= self.pool2(x)



      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      x = self.act5(x)
      x = self.drop(x)
      x = self.classifier2(x)
      return x #output
