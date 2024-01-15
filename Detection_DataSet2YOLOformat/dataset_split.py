import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(dataset_path, output_path, train_size=0.9, random_seed=42):
    # 创建输出文件夹
    os.makedirs(output_path, exist_ok=True)

    # 获取所有图像和标签文件的路径
    images_dir = os.path.join(dataset_path, 'images')
    label_dir = os.path.join(dataset_path, 'labels')
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    # 使用sklearn库的train_test_split函数进行划分
    train_images, val_images, train_labels, val_labels = train_test_split(
        image_files, label_files, train_size=train_size, random_state=random_seed
    )

    # 创建训练集和验证集的文件夹
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # 复制图像文件到训练集和验证集的文件夹
    for image_file in train_images:
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(train_path, image_file))
    for image_file in val_images:
        shutil.copy(os.path.join(images_dir, image_file), os.path.join(val_path, image_file))

    # 创建训练集和验证集标签的文件夹
    train_label_path = os.path.join(output_path, 'train_label')
    val_label_path = os.path.join(output_path, 'val_label')
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    # 复制标签文件到训练集和验证集标签的文件夹
    for label_file in train_labels:
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_path, label_file))
    for label_file in val_labels:
        shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_path, label_file))


if __name__ == "__main__":
    dataset_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\Dataset"
    output_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\YOLO_dataset"

    split_dataset(dataset_path, output_path)
