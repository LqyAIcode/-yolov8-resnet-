import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset
import json
import os

# 自定义数据集类
class RegressionDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None, additional_transform=None):
        self.image_folder = image_folder
        self.labels = self.load_labels(label_file)
        self.transform = transform
        self.additional_transform = additional_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels[idx]["name"]
        img_path = os.path.join(self.image_folder, img_name)
        img = Image.open(img_path).convert('RGB')

        label = float(self.labels[idx]["age"])
        if 1 <= label <= 15 and self.additional_transform:
            img = self.additional_transform(img)
        elif self.transform:
            img = self.transform(img)

        return img, label

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels


# 定义 ResNet 模型
class ResNetRegression(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNetRegression, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
if __name__ =='__main__':
    # 数据预处理和标准化
    transform = transforms.Compose([


        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()

    ])
    # 在训练集上应用额外的数据增强
    additional_transform = transforms.Compose([

        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])

    # 训练集和测试集路径及标签文件
    # train_folder = 'output_path/dataset_cls/train_cls'
    # test_folder = 'output_path/dataset_cls/test_cls'
    # train_folder = 'output_path/dataset_raw/trainval'
    # test_folder = 'output_path/dataset_raw/test'
    train_folder = 'C:\\Users\\Administrator\\PycharmProjects\\conda_pythonProject4\\pathtest\\train_cls_2'
    test_folder = 'C:\\Users\\Administrator\\PycharmProjects\\conda_pythonProject4\\pathtest\\test_cls_2'
    train_label_file = 'output_path/dataset_cls/trainval_labels.json'
    test_label_file = 'output_path/dataset_cls/test_labels.json'

    # 创建训练集和测试集数据集实例
    train_dataset = RegressionDataset(train_folder, train_label_file, transform=transform,additional_transform=additional_transform)
    test_dataset = RegressionDataset(test_folder, test_label_file, transform=transform)

    # 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型、损失函数和优化器
    model = ResNetRegression()
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 在训练集上训练模型
    num_epochs = 80
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {average_loss}")

        # 在测试集上评估模型
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss += criterion(outputs, labels.view(-1, 1)).item()

            average_test_loss = test_loss / len(test_loader)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {average_test_loss}")
    torch.save(model.state_dict(), 'resnet_model_regression_7_test.pth')

    # 在测试集上评估模型
    criterion = nn.L1Loss()
    model.eval()
    total_absolute_error = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.float().view(-1, 1))
            total_absolute_error += loss.item() * len(labels)
            num_samples += len(labels)

    # 计算平均绝对误差 (MAE)
    mae = total_absolute_error / num_samples
    print(f"Mean Absolute Error (MAE) on Test Set: {mae}")
