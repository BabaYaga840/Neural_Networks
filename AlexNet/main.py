import torch
from torchstat import stat
import torchvision
from AlexNet import AlexNet
import torchvision.transforms as transforms
from Train_test import Training

num_epochs=10
learning_rate=0.001
image_resolution=224
n_classes=10


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((image_resolution, image_resolution)), transforms.Normalize((0.1307,), (0.3081,))])
path="/content/drive/MyDrive/AlexNet/data" #data
                                            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root=path, train=True,transform = transform,download=True)
test_dataset = torchvision.datasets.MNIST(root=path, train=False,transform = transform,download=True)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size=self.batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=self.batch_size,shuffle=True)

input_channel = next(iter(train_dataloader))[0].shape[1]

model = AlexNet(input_channel=input_channel, n_classes=n_classes).to(device)

trainer = Training(model=model, optimizer=sgd, learning_rate=learning_rate, 
train_dataloader=train_dataloader, num_epochs=num_epochs,test_dataloader=test_dataloader)
trainer.runner()
