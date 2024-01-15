import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from resnet_jsonload_train import RegressionDataset,ResNetRegression
import os
# class RegressionDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.file_paths, self.labels = self._load_data()
#
#     def _load_data(self):
#         file_paths = []
#         labels = []
#         for label in os.listdir(self.root_dir):
#             label_dir = os.path.join(self.root_dir, label)
#             if os.path.isdir(label_dir):
#                 for file_name in os.listdir(label_dir):
#                     file_path = os.path.join(label_dir, file_name)
#                     file_paths.append(file_path)
#                     labels.append(float(label))  # 将文件夹名（标签）转换为浮点数
#
#         return file_paths, labels
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, index):
#         image = Image.open(self.file_paths[index]).convert("RGB")
#
#         if self.transform:
#             image = self.transform(image)
#
#         label = torch.tensor(self.labels[index], dtype=torch.float32)
#
#         return image, label
# class ResNetRegression(nn.Module):
#     def __init__(self):
#         super(ResNetRegression, self).__init__()
#         resnet = models.resnet18()
#         # 去掉ResNet的最后一层（分类层）
#         self.resnet = nn.Sequential(*list(resnet.children())[:-1])
#         # 添加自定义的全连接层，用于回归
#         self.fc = nn.Linear(512, 1)
#         # self.fc2 = nn.Linear(100, 1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         # x = self.fc2(x)
#         x = self.sigmoid(x)*100
#         return x



# 数据预处理和标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建测试集数据集实例
# test_folder = 'output_path/dataset_cls/test_cls'
test_folder = 'pathtest/test_cls_2'
test_label_file = 'output_path/dataset_cls/test_labels.json'
test_dataset = RegressionDataset(test_folder, test_label_file, transform=transform)

# 创建 DataLoader
batch_size = 32
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetRegression()  # 替换为你的模型定义
model.load_state_dict(torch.load('resnet_model_regression_3.pth'))  # 加载训练好的模型权重
model.to(device)
model.eval()

# 定义损失函数
criterion = nn.L1Loss()

# 在测试集上评估模型
total_absolute_error = 0.0
num_samples = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # 如果使用 GPU，将数据移到 GPU 上
        outputs = model(images)
        print(labels)
        print(outputs)
        loss = criterion(outputs, labels.view(-1, 1))
        total_absolute_error += loss.item() * len(labels)
        num_samples += len(labels)

# 计算平均绝对误差 (MAE)
mae = total_absolute_error / num_samples
print(f"Mean Absolute Error (MAE) on Test Set: {mae}")

