from PIL import Image
from ultralytics import YOLO
import torch
import os
import json

def clear_output_folder(output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输出文件夹中所有文件
    files = os.listdir(output_folder)

    # 删除输出文件夹中的所有文件
    for file in files:
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)


def get_closest_box(boxes, target_box):
    # 计算每个检测框与目标框的中心点之间的距离
    target_x, target_y = (target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2
    distances = [((box[0] + box[2]) / 2 - target_x) ** 2 + ((box[1] + box[3]) / 2 - target_y) ** 2 for box in boxes]
    print(distances)
    # 找到距离最小的检测框的索引
    closest_index = min(range(len(distances)), key=distances.__getitem__)

    return boxes[closest_index]


def crop_and_save_image(image, coordinates, output_path, scale_adjust=0.2):
    # 打开图像
    # image = Image.open(image_path)
    w, h = image.size
    # 使用张量索引提取坐标
    left, top, right, bottom = coordinates
    width = right - left
    height = bottom - top
    left = left - scale_adjust * width
    top = top - 1.5 * scale_adjust * height
    right = right + scale_adjust * width
    bottom = bottom + 0.5 * scale_adjust * height
    left = left
    top = top
    right = right
    bottom = bottom
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > w:
        right = w
    if bottom > h:
        bottom = h
    # 使用坐标裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    # 保存裁剪后的图像
    cropped_image.save(output_path)


def main(input_folder, output_folder, model, scale_adjust=0.2):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    # 清空之前的文件夹
    clear_output_folder(output_folder)
    # 获取输入文件夹中所有的 JPEG 图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    # 读取 JSON 数据
    with open(label_path, "r") as f:
        label_data = json.load(f)
    # 预先加载模型
    for image_file in image_files:
        # 构建输入图像路径和输出图像路径
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # 打开图像
        image = Image.open(input_image_path)
        print(type(image))
        if image.mode == 'RGBA' or 'LA':
            image = image.convert('RGB')

        # 使用 YOLO 模型获取坐标
        results = model(input_image_path)
        # 判断是否有目标检测结果
        for r in results:
            print(r.boxes.xyxy)
            if torch.numel(r.boxes.xyxy) > 0:
                # 获取目标框的坐标（假设你有一个函数可以获取标签框的坐标，比如 get_target_box）
                target_box = get_target_box(image_file,label_data)  # 需要替换成实际获取标签框坐标的逻辑

                # 找到与标签框最接近的检测框
                closest_box = get_closest_box(r.boxes.xyxy.tolist(), target_box)

            else:
                w, h = image.size
                closest_box = [0, 0, w, h]
        # 裁剪并保存图像
        crop_and_save_image(image, closest_box, output_image_path, scale_adjust)


def get_target_box(image_file, label_data):
    # 在 label_data 中找到与 image_file 对应的条目
    for entry in label_data:
        if entry["name"] == image_file:
            bbox = entry["bbox"]
            return [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]]

    # 如果找不到对应的条目，可以返回一个默认值或者抛出异常，具体根据需求来定
    return [0, 0, 0, 0]



# 路径

# #生成训练集crop人脸
# label_path ='output_path/dataset_cls/trainval_labels.json'
# input_folder = "C:\\Users\\Administrator\\Desktop\\Dataset\\images"
# ## output_folder = "pathtest\\train_cls_2"  # 保存裁剪后图像的路径
# output_folder = "pathtest2\\train_cls_2"  # 保存裁剪后图像的路径
# # 生成测试集集crop人脸
label_path ='output_path/dataset_cls/test_labels.json'
input_folder = "C:\\Users\\Administrator\\Desktop\\Dataset\\test"
##output_folder = "pathtest\\test_cls_2"  # 保存裁剪后图像的路径
output_folder = "pathtest2\\test_cls_2"  # 保存裁剪后图像的路径

# 加载预训练的YOLOv8n模型
model = YOLO('runs/detect/train53_epoch/weights/best.pt')
main(input_folder, output_folder, model)
