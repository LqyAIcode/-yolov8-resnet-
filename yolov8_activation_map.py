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

class MyConcat(nn.Module):
    def __init__(self, dimension=1):
        super(MyConcat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def random_num(size,end):
    range_ls=[i for i in range(end)]
    num_ls=[]
    for i in range(size):
        num=random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls

# 加载YOLOv8模型
model = YOLO('runs/detect/train53_epoch/weights/best.pt')
img_path = r"C:\Users\Administrator\Desktop\Dataset\images\trainval679.jpg"
image = Image.open(img_path).convert('RGB')
model = model.model.model
print(model)
# model = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name in model.named_children():
    print(name[0])
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((640,640))]
)
image = transform(image).unsqueeze(0)
new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'0': '0', '1': '1','2':'2','3':'3','4':'4','5':'5','6':'6','7':'7','8':'8','9':'9','10': '10'})
# 使用 MyConcat 替代原来的 Concat 层

# new_model._modules[11] = MyConcat()
print(new_model)
out = new_model(image)
tensor_ls=[(k,v) for  k,v in out.items()]
v=tensor_ls[10][1]
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
# # 定义一个字典来存储激活图

# activations = {}
#
# # 定义一个钩子函数来截获激活值
# def hook_fn(module, input, output):
#     activations[module] = output
#
# # 注册钩子到第二层
# target_layer = model.model.model[1]
# target_layer.register_forward_hook(hook_fn)
#
# # 进行预处理
# preprocess = transforms.Compose([
#     transforms.Resize((416, 416)),
#     transforms.ToTensor(),
# ])
# input_tensor = preprocess(image).unsqueeze(0)
#
# # 将输入数据传递给模型
# with torch.no_grad():
#     output = model.model.model(input_tensor)
#
# # 可视化第二层的激活图
# activation_map = activations[target_layer].squeeze(0).mean(dim=0).cpu().numpy()
# plt.imshow(activation_map, cmap='viridis')
# plt.title("Layer 2 Activation Map")
# plt.show()
