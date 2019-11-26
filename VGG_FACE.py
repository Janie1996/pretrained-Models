
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
from functools import reduce
from torch.autograd import Variable
from torchvision import datasets,transforms
import torch.optim as optim


class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))

VGG_FACE = nn.Sequential( # Sequential,
    nn.Conv2d(3,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(64,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
    nn.ReLU(),
    nn.MaxPool2d((2, 2),(2, 2),(0, 0),ceil_mode=True),
    Lambda(lambda x: x.view(x.size(0),-1)), # View,
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(25088,4096)), # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,4096)), # Linear,
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(4096,2622)), # Linear,
    nn.Linear(2622,7),
    nn.Softmax(),
)


#Pretrained
model_dict = VGG_FACE.state_dict()
vggface_state_dict = torch.load("VGG_FACE.pth")
pretrained_dict = {k: v for k, v in vggface_state_dict.items() if k in model_dict and not k.startswith("38")}
model_dict.update(pretrained_dict)
VGG_FACE.load_state_dict(model_dict)
print("test")

# x=torch.ones(1,3,224,224)
# y=net(x)
# print(y)

data_transform = transforms.Compose([

    transforms.Resize(224),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='E:/fer2013/train/', transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = datasets.ImageFolder(root='E:/fer2013/test/', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

net=VGG_FACE.cuda()
net.train()
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

for epoch in range(30):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        inputs=inputs.cuda()
        labels=labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = cirterion(outputs, labels)
        loss.backward()
        optimizer.step()

    total=0
    correct=0
    for j, tdata in enumerate(test_loader, 0):
        net.eval()
        x, y = tdata
        x=x.cuda()
        y=y.cuda()
        out=net(x)
        _, predicted = torch.max(out.data, 1)
        total =total+ y.size(0)
        correct = correct+(predicted == y).sum()
    print('测试分类准确率为：%.3f%%' % (100 * correct / total))

