import json
import cv2

def json_to_yolo(json_file, output_path,images_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)
    for entry in data:


        # 获取图像名、类别标签和边界框坐标
        image_name = entry['name']
        class_label = entry['age']
        bbox = entry['bbox']
        print(image_name.split)
        output_path_write = f"{output_path}\\{image_name}"
        with open(output_path_write, 'w') as f_out:
            # 加载图像获取原始宽度和高度
            image_path = f"{images_dir}\\{image_name}"
            img = cv2.imread(image_path)
            image_height, image_width, _ = img.shape

            # 将边界框坐标转换为YOLO格式的相对坐标
            x_center = (bbox['x1'] + bbox['x2']) / 2.0/image_width
            y_center = (bbox['y1'] + bbox['y2']) / 2.0/image_height
            width = (bbox['x2'] - bbox['x1'])/image_width
            height = (bbox['y2'] - bbox['y1'])/image_height

            # 将信息写入文本文件
            line = f"{class_label} {x_center} {y_center} {width} {height} # {image_name}\n"
            f_out.write(line)

if __name__ == "__main__":
    json_file_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\test_labels.json"
    output_txt_path = f"C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\testlabel"
    images_dir_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\test"
    json_to_yolo(json_file_path, output_txt_path,images_dir_path)
