import json
import cv2
import os

def json_to_yolo(json_file, output_dir, images_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    for entry in data:
        # 获取图像名、类别标签和边界框坐标
        image_name = entry['name']
        class_label = entry['age']
        bbox = entry['bbox']
        boxlabel = 0#只需要检测出人脸，因此只需要一个label
        # 加载图像获取原始宽度和高度
        image_path = os.path.join(images_dir, image_name)
        img = cv2.imread(image_path)
        image_height, image_width, _ = img.shape

        # 将边界框坐标转换为YOLO格式的相对坐标
        x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        # 将信息写入单独的文本文件
        print(os.path.splitext(image_name)[0])
        output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".txt")
        with open(output_path, 'w') as f_out:
            line = f"{boxlabel} {x_center} {y_center} {width} {height}\n"
            f_out.write(line)

if __name__ == "__main__":
    # #训练集标签
    # json_file_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\trainval_labels.json"
    # output_txt_dir = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\trainlabel"
    # images_dir = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\trainval"

    json_file_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\test_labels.json"
    output_txt_dir = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\testlabel"
    images_dir = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\test"
    # 确保输出目录存在
    os.makedirs(output_txt_dir, exist_ok=True)

    json_to_yolo(json_file_path, output_txt_dir, images_dir)