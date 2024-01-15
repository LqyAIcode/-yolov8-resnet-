import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Microsoft YaHei'  # 设置字体为微软雅黑


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.labels = self.load_labels(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels[idx]["name"]
        img_path = os.path.join(self.image_folder, img_name)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # 使用原始数据集的标签作为回归任务的标签
        regression_label = float(self.labels[idx]["age"])

        # 将原始数据集的标签转化为分类任务的标签
        if 1 <= regression_label <= 30:
            classification_label = 0
        if 31 <= regression_label <= 100:
            classification_label = 1
        # elif 21 <= regression_label <= 40:
        #     classification_label = 1
        # elif 41 <= regression_label <= 60:
        #     classification_label = 2
        # else:
        #     classification_label = 3

        return img, {"regression_label": regression_label, "classification_label": classification_label}

    def load_labels(self, label_file):
        with open(label_file, 'r') as f:
            labels = json.load(f)
        return labels


# 定义分类和回归任务的模型
class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        resnet = models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = resnet.fc.in_features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
class ResNeXtClassification(nn.Module):
    def __init__(self, num_classes):
        super(ResNeXtClassification, self).__init__()
        resnext = models.resnext50_32x4d(pretrained=True)  # You can choose different configurations
        # Modify the final fully connected layer for the number of output classes
        resnext.fc = nn.Linear(resnext.fc.in_features, num_classes)
        self.resnext = resnext

    def forward(self, x):
        return self.resnext(x)

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


if __name__ == '__main__':
    # 数据预处理和标准化
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5),
        transforms.ToTensor()
    ])
    transform_cls = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # 训练集和测试集路径及标签文件
    train_folder = 'C:\\Users\\Administrator\\PycharmProjects\\conda_pythonProject4\\pathtest\\train_cls_2'
    test_folder = 'C:\\Users\\Administrator\\PycharmProjects\\conda_pythonProject4\\pathtest\\test_cls_2'
    train_label_file = 'output_path/dataset_cls/trainval_labels.json'
    test_label_file = 'output_path/dataset_cls/test_labels.json'

    # 创建训练集和测试集数据集实例
    train_dataset = CustomDataset(train_folder, train_label_file, transform=transform_test)
    test_dataset = CustomDataset(test_folder, test_label_file, transform=transform_test)

    # 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化分类模型、回归模型、损失函数和优化器
    classification_model = ClassificationModel(num_classes=2)
    # classification_model = ResNeXtClassification(num_classes=4)
    regression_model = RegressionModel()

    regression_model.load_state_dict(torch.load('resnet_model_regression_7_selected_augmentation.pth'))
    classification_model.load_state_dict(torch.load('classification_model_4.pth'))

    classification_model.to(device)
    regression_model.to(device)
    # 为分类交叉熵损失函数设置类别权重
    class_weights = torch.tensor([4.0, 1.0], dtype=torch.float32).to(device)
    classification_criterion = nn.CrossEntropyLoss(weight=class_weights)
    # classification_criterion = nn.CrossEntropyLoss()
    regression_criterion = nn.L1Loss()

    classification_optimizer = torch.optim.Adam(classification_model.parameters(), lr=0.001)
    regression_optimizer = torch.optim.Adam(regression_model.parameters(), lr=0.001)

    # 在测试集上评估模型
    classification_model.eval()
    regression_model.eval()
    total_classification_correct = 0
    total_regression_absolute_error = 0.0
    num_samples = 0
    classification_preds_list = []
    classification_labels_list = []
    plt.figure(figsize=(8, 6))

    with torch.no_grad():
        for images, labels in test_loader:
            images, regression_labels, classification_labels = images.to(device), labels["regression_label"].to(device), labels["classification_label"].to(device)

            # 分类模型的预测结果
            classification_outputs = classification_model(images)
            _, classification_preds = torch.max(classification_outputs, 1)
            classification_preds_list.extend(classification_preds.cpu().numpy())
            classification_labels_list.extend(classification_labels.cpu().numpy())
            # 获取分类结果对应的区间范围
            # classification_ranges = [(1.0, 20.0), (21.0, 40.0), (41.0, 60.0), (61.0, 100.0)]
            classification_ranges = [(1.0, 30.0), (31.0, 100.0)]
            classification_intervals = [(min_val, max_val) for (min_val, max_val) in classification_ranges]
            classification_intervals = torch.tensor(classification_intervals, dtype=torch.float32, device=device)

            # 根据分类结果找到对应区间的最大值和最小值
            max_values = torch.gather(classification_intervals[:, 1], 0, classification_preds)
            min_values = torch.gather(classification_intervals[:, 0], 0, classification_preds)

            # 如果回归结果与分类结果一致，则直接输出回归结果
            # consistent_mask = classification_preds == classification_labels
            regression_outputs = regression_model(images)
            consistent_mask = (regression_outputs.view(-1) >= min_values) & (regression_outputs.view(-1) <= max_values)
            consistent_regression_preds = regression_outputs.view(-1)[consistent_mask]

            # 如果不一致，则判定回归结果与分类结果区间的关系
            inconsistent_mask = ~consistent_mask
            inconsistent_classification_labels = classification_labels[inconsistent_mask]
            inconsistent_regression_outputs = regression_outputs.view(-1)[inconsistent_mask]
            print(max_values[inconsistent_mask],len(max_values[inconsistent_mask]))
            # 判定回归输出与区间的关系
            final_regression_preds = torch.zeros_like(regression_labels, dtype=consistent_regression_preds.dtype)
            final_regression_preds[consistent_mask] = consistent_regression_preds
            final_regression_preds = torch.clamp(regression_outputs.view(-1), min_values, max_values)


            print(consistent_mask)
            print(regression_labels)
            print(final_regression_preds)
            # 计算分类任务的准确度
            total_classification_correct += torch.sum(classification_preds == classification_labels).item()

            # 计算回归任务的平均绝对误差 (MAE)
            regression_absolute_error = torch.abs(final_regression_preds - regression_labels.view(-1))
            total_regression_absolute_error += torch.sum(regression_absolute_error).item()

            num_samples += len(classification_labels)
            plt.scatter(regression_labels.cpu().numpy(), final_regression_preds.cpu().numpy())

    # 计算分类任务的准确度和回归任务的平均绝对误差 (MAE)
    classification_accuracy = total_classification_correct / num_samples
    regression_mae = total_regression_absolute_error / num_samples

    print(
        f"Classification Accuracy on Test Set: {classification_accuracy}, Regression MAE on Test Set: {regression_mae}")
    # 计算分类准确度
    classification_accuracy = accuracy_score(classification_labels_list, classification_preds_list)
    print(f"Classification Accuracy on Test Set: {classification_accuracy}")


    # 可视化回归结果散点图

    plt.xlabel('实际回归标签')
    plt.ylabel('预测回归输出')
    plt.title('回归结果：实际 vs 预测')
    plt.show()

    # 可视化混淆矩阵
    conf_matrix = confusion_matrix(classification_labels_list, classification_preds_list)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


