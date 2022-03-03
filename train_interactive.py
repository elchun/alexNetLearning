from json import load
from pickletools import optimize
from alexNet import AlexNet
import torch

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torchvision import transforms
from torchvision.utils import save_image

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

import os, os.path as osp
import code

def load_data():
    """
    Load data from CIFAR dataset
    returns train_dl, val_dl, test_dl
    """
    tranform_train = transforms.Compose([transforms.Resize((227,227)), transforms.RandomHorizontalFlip(p=0.7), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    tranform_test = transforms.Compose([transforms.Resize((227,227)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    train_ds = CIFAR10("data/", train=True, download=True, transform=tranform_train) #40,000 original images + transforms

    val_size = 10000 #there are 10,000 test images and since there are no transforms performed on the test, we keep the validation as 10,000
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size]) #Extracting the 10,000 validation images from the train set
    test_ds = CIFAR10("data/", train=False, download=True, transform=tranform_test) #10,000 images

    #passing the train, val and test datasets to the dataloader
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
    return train_dl, val_dl, test_dl


class Trainer():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AlexNet()
        self.model = self.model.to(device=self.device)

        self.lr = 1e-4
        self.load_model = True
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.weight_path = "model_weights/"
    
    def train(self, train_data, val_data, num_epochs=50):
        for epoch in range(num_epochs):
            loss_ep = 0

            # Train
            for batch_idx, (data, targets) in enumerate(train_data):
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)

                # Forward
                self.optimizer.zero_grad()
                predicted = self.model(data)
                loss = self.loss_fn(predicted, targets)
                loss.backward()
                self.optimizer.step()
                loss_ep += loss.item() # Extract loss as float
                if batch_idx % 5 == 0:
                    print(f'Batch idx: {batch_idx}, batch loss: {loss.item()}')
            
            print(f'Loss for epoch: {epoch} :::: {loss_ep / len(train_data)}')

            # Validate
            with torch.no_grad():
                num_correct = 0
                num_samples = 0
                for batch_idx, (data, targets) in enumerate(val_data):
                    data = data.to(device=self.device)
                    targets = targets.to(device=self.device)

                    predicted = self.model(data)
                    _, predictions = predicted.max(1)
                    num_correct += (predictions == targets).sum()
                    num_samples += predictions.size(0)
                print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')
        
    def save_weights(self):

        i = 0
        weight_fn = osp.join(self.weight_path, f"weights{i}")
        while osp.exists(weight_fn):
            i += 1
            weight_fn = osp.join(self.weight_path, f"weights{i}")
        torch.save(self.model.state_dict(), weight_fn)
        print("Model weights saved")



if __name__ == "__main__":
    torch.manual_seed(43)
    train_dl, val_dl, test_dl = load_data()
    trainer = Trainer()
    code.interact(local=locals())
    # trainer.train(train_dl, val_dl)
    # trainer.save_weights()


