# from PIL import Image
# from ultralytics import YOLO
# import torch
# import os
#
# def crop_and_save_image(image_path, coordinates, output_path,scale_adjust=0.2):
#     # 打开图像
#     image = Image.open(image_path)
#     w,h = image.size
#     # 使用张量索引提取坐标
#     left, top, right, bottom = coordinates
#     width = right - left
#     height = bottom - top
#     left = left - scale_adjust*width
#     top = top - 1.5*scale_adjust*width
#     right = right+ scale_adjust * width
#     bottom = bottom + 0.5*scale_adjust*width
#     if left<0:
#         left = 0
#     if top<0:
#         top = 0
#     if right>w:
#         right = w
#     if bottom >h:
#         bottom = h
#     # 使用坐标裁剪图像
#     cropped_image = image.crop((left, top, right, bottom))
#     # 保存裁剪后的图像
#     cropped_image.save(output_path)
#
# #路径
# image_path = "D:\\ultralytics-main\\ultralytics\\assets\\zidane.jpg"  # 你的图像路径
# output_path = "pathtest/image.jpg"  # 保存裁剪后图像的路径
#
# # 加载预训练的YOLOv8n模型
# model = YOLO('runs/detect/train4_epoch100_pretrain/weights/last.pt')
#
# # 在'bus.jpg'上运行推理
# results = model(image_path)  # 结果列表
#
# # 展示结果
# for r in results:
#     print(r.boxes.xyxy[0])
#     im_array = r.plot()  # 绘制包含预测结果的BGR numpy数组
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL图像
#     im.show()  # 显示图像
#     # im.save('results.jpg')  # 保存图像
#     coordinates = r.boxes.xyxy[0].tolist()  # left, top, right, bottom
#     crop_and_save_image(image_path, coordinates, output_path,0.2)



from PIL import Image
from ultralytics import YOLO
import torch
import os

def clear_output_folder(output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 获取输出文件夹中所有文件
    files = os.listdir(output_folder)

    # 删除输出文件夹中的所有文件
    for file in files:
        file_path = os.path.join(output_folder, file)
        os.remove(file_path)
def crop_and_save_image(image, coordinates, output_path,scale_adjust=0.2):
    # 打开图像
    # image = Image.open(image_path)
    w,h = image.size
    # 使用张量索引提取坐标
    left, top, right, bottom = coordinates
    width = right - left
    height = bottom - top
    left = left - scale_adjust*width
    top = top - 1.5*scale_adjust*width
    right = right+ scale_adjust * width
    bottom = bottom + 0.5*scale_adjust*width
    if left<0:
        left = 0
    if top<0:
        top = 0
    if right>w:
        right = w
    if bottom >h:
        bottom = h
    # 使用坐标裁剪图像
    cropped_image = image.crop((left, top, right, bottom))
    # 保存裁剪后的图像
    cropped_image.save(output_path)

def main(input_folder, output_folder, model,scale_adjust=0.2):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    #清空之前的文件夹
    clear_output_folder(output_folder)
    # 获取输入文件夹中所有的 JPEG 图片文件
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    # 预先加载模型
    for image_file in image_files:
        # 构建输入图像路径和输出图像路径
        input_image_path = os.path.join(input_folder, image_file)
        output_image_path = os.path.join(output_folder, image_file)

        # 打开图像
        image = Image.open(input_image_path)
        print(type(image))
        if image.mode=='RGBA'or'LA':
            image = image.convert('RGB')
        # 使用 YOLO 模型获取坐标
        results = model(input_image_path)  # 这里需要替换成你实际的 YOLO 模型推理逻辑
        for r in results:
            # print(r.boxes.xyxy)
            if torch.numel( r.boxes.xyxy ) == 0:
                w,h = image.size
                coordinates = [0,0,w,h]
            else:
                coordinates = r.boxes.xyxy[0].tolist()

        # 裁剪并保存图像
        crop_and_save_image(image, coordinates, output_image_path,scale_adjust)
#路径
# input_folder = "D:\\ultralytics-main\\ultralytics\\assets"  # 图像路径
# #生成训练集crop人脸
# input_folder = "C:\\Users\\Administrator\\Desktop\\Dataset\\images"
# output_folder = "pathtest\\train_cls"  # 保存裁剪后图像的路径
#生成测试集集crop人脸
input_folder = "C:\\Users\\Administrator\\Desktop\\Dataset\\test"
output_folder = "pathtest\\test_cls"  # 保存裁剪后图像的路径

# 加载预训练的YOLOv8n模型
model = YOLO('runs/detect/train4_epoch100_pretrain/weights/last.pt')
main(input_folder, output_folder, model)
