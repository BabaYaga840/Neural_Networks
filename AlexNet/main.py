import torch
from torchstat import stat
import torchvision
from AlexNet import AlexNet
import torchvision.transforms as transforms
from Train_test import Training
import yaml

config_file = open("/content/drive/MyDrive/AlexNet/config.yaml", 'r')
config = yaml.safe_load(config_file)
  
num_epochs=config['parameters']['num_epochs']
learning_rate=config['parameters']['learning_rate']
image_resolution=config['parameters']['image_resolution']
n_classes=config['parameters']['n_classes']
batch_size=config['parameters']['batch_size']

transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((image_resolution, image_resolution)), transforms.Normalize((0.1307,), (0.3081,))])
path="/content/drive/MyDrive/AlexNet/data" #data
                                            
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root=path, train=True,transform = transform,download=True)
test_dataset = torchvision.datasets.MNIST(root=path, train=False,transform = transform,download=True)
train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset,batch_size=batch_size,shuffle=True)

input_channel = next(iter(train_dataloader))[0].shape[1]

model = AlexNet(input_channel=input_channel, n_classes=n_classes).to(device)

trainer = Training(model=model, learning_rate=learning_rate, 
train_dataloader=train_dataloader, num_epochs=num_epochs,test_dataloader=test_dataloader)
trainer.runner()
