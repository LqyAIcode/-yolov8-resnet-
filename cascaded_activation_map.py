from PIL import Image
from ultralytics import YOLO
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names


def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

#加载回归模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        resnet = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# 加载YOLOv8模型
regression_model = RegressionModel()
regression_model.load_state_dict(torch.load('resnet_model_regression_7_selected_augmentation.pth'))
img_path = r"C:\Users\Administrator\PycharmProjects\conda_pythonProject4\pathtest2\test_cls_2\test26.jpg"
image = Image.open(img_path).convert('RGB')
regression_model = regression_model.features
print(regression_model)
# model = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name in regression_model.named_children():
    print(name[0])
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
image = transform(image).unsqueeze(0)
new_model = torchvision.models._utils.IntermediateLayerGetter(regression_model, {'0': '0', '1': '1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8'})

print(new_model)
out = new_model(image)
tensor_ls=[(k,v) for  k,v in out.items()]
v=tensor_ls[7][1]
#取消Tensor的梯度并转成三维tensor，否则无法绘图
v=v.data.squeeze(0)

print(v.shape)  # torch.Size([512, 28, 28])

#随机选取25个通道的特征图
channel_num = random_num(25,v.shape[0])
print(channel_num)
# channel_num = []
plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index+1,)
    plt.imshow(v[channel, :, :])
plt.savefig("feature.jpg",dpi=300)
plt.show()