from os import TMP_MAX
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CyclicLR
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

config_file = open("/content/drive/MyDrive/AlexNet/config.yaml", 'r')
config = yaml.safe_load(config_file)

learning_rate=config['parameters']['learning_rate']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                                       
class Training:
    
    def __init__(self, model, learning_rate, train_dataloader, num_epochs, 
                test_dataloader):
        self.model = model
        self.learning_rate = learning_rate
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_epochs = num_epochs

     
    def plot(train_accu,test_accu,train_losses,test_losses):
        Path('/content/drive/MyDrive/AlexNet/plot').mkdir(parents=True, exist_ok=True)
        plot1 = plt.figure(1)
        plt.plot(train_accu, '-o')
        plt.plot(test_accu, '-o')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['Train','Test'])
        plt.title('Train vs Test Accuracy')            
        plt.savefig('/content/drive/MyDrive/AlexNet/plot/plot_train_test_acc.png')
            
        plot2 = plt.figure(2)
        plt.plot(train_losses,'-o')
        plt.plot(test_losses,'-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train','Test'])
        plt.title('Train vs Test Losses')
        plt.savefig('/content/drive/MyDrive/AlexNet/plot/plt_train_test_loss.png')
        
    def runner(self):
        best_accuracy = float('-inf')
        criterion = nn.CrossEntropyLoss()
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        scheduler = CyclicLR(self.optimizer, base_lr=1e-07, max_lr=0.1, step_size_up=100, mode="triangular")
       
        
        train_losses = []
        train_accu = []
        test_losses = []
        test_accu = []
        # Train 
        total_step = len(self.train_dataloader)
        for epoch in range(self.num_epochs):
            running_loss = 0
            correct = 0
            total = 0
            for i, (images, labels) in enumerate(self.train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                
                # Forward 
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_loss=running_loss/len(self.train_dataloader)
                train_accuracy = 100.*correct/total
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.3f}, Train Loss: {:.4f}'
                    .format(epoch+1, self.num_epochs, i+1, total_step, train_accuracy, loss.item()))
                
            #evaluation           
            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                running_loss = 0
                for images, labels in self.test_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = self.model(images)
                    loss= criterion(outputs,labels)
                    running_loss+=loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    test_loss=running_loss/len(self.test_dataloader)
                    test_accuracy = (correct*100)/total
                print('Epoch: %.0f | Test Loss: %.3f | Accuracy: %.3f'%(epoch+1, test_loss, test_accuracy))
            scheduler.step()

            #model save
            if test_accuracy > best_accuracy:
                best_accuracy=test_accuracy
                Path('/content/drive/MyDrive/AlexNet/model_store').mkdir(parents=True, exist_ok=True)
                torch.save(self.model.state_dict(), 'model_store-best-model-parameters.pt')
            
            #checkpoint save
            path = 'checkpoints/checkpoint{:04d}.pth.tar'.format(epoch)
            Path('checkpoints/').mkdir(parents=True, exist_ok=True)
            torch.save(
                    {
                        'epoch': self.num_epochs,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss
                    }, path)

            train_accu.append(train_accuracy)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            test_accu.append(test_accuracy)
        #Plot
        plot(train_accu,test_accu,train_losses,test_losses)
       
                                                                                                      
