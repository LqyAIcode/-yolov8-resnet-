import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 自定义回归数据集类
class RegressionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths, self.labels = self._load_data()

    def _load_data(self):
        file_paths = []
        labels = []
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    file_paths.append(file_path)
                    labels.append(float(label))  # 将文件夹名（标签）转换为浮点数

        return file_paths, labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = Image.open(self.file_paths[index]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return image, label

# 数据预处理和标准化
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 创建数据集实例
train_dataset = RegressionDataset(root_dir='D:\\ultralytics-main\\Dataset_cls_sorted\\train', transform=transform)
test_dataset = RegressionDataset(root_dir='D:\\ultralytics-main\\Dataset_cls_sorted\\test', transform=transform)

# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义回归模型
class ResNetRegression(nn.Module):
    def __init__(self):
        super(ResNetRegression, self).__init__()
        resnet = models.resnet18(pretrained=True)
        # 去掉ResNet的最后一层（分类层）
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # 添加自定义的全连接层，用于回归
        self.fc = nn.Linear(512, 1)
        # self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.fc2(x)
        x = self.sigmoid(x)*100
        return x
if __name__=='__main__':
    # 初始化模型、损失函数和优化器
    model = ResNetRegression()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()

        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # print(labels)
                # print(outputs)
                test_loss += criterion(outputs, labels.view(-1, 1)).item()

            average_test_loss = test_loss / len(test_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item()}, Test Loss: {average_test_loss}")
    torch.save(model.state_dict(), 'resnet_model_regression_2.pth')