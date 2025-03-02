import torch
from torch import nn, save,load
import torchvision
from torchvision.transforms import ToTensor
import torch.optim as optim
from PIL import Image

train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())

dataset = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x): 
        return self.model(x)

device = torch.device("mps")
clf = Classifier().to(device)
opt = optim.Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__": 
    for epoch in range(5):
        for batch in dataset: 
            X,y = batch 
            X, y = X.to(device), y.to(device) 
            y_hat = clf(X) 
            loss = loss_fn(y_hat, y) 

            opt.zero_grad()
            loss.backward() 
            opt.step() 

        print(f"Epoch:{epoch} loss is {loss.item()}")

with open('model.pt', 'wb') as f: 
        save(clf.state_dict(), f)


"""
 with open('model.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  

    img = Image.open('0.jpg') 
    img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(torch.argmax(clf(img_tensor)))
"""