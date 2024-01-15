import yaml

def create_dataset_yaml(train_path, val_path, class_names, yaml_path):
    dataset_yaml = {
        'train': train_path,
        'val': val_path,
        'nc': len(class_names),
        'names': class_names
    }

    with open(yaml_path, 'w') as yaml_file:
        yaml.dump(dataset_yaml, yaml_file, default_flow_style=False)

if __name__ == "__main__":
    train_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\YOLO_dataset\\images\\train"
    val_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\YOLO_dataset\\images\\val"
    class_names = ['Face']
    # class_names = ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'age7', 'age8', 'age9', 'age10',
    #                'age11', 'age12', 'age13', 'age14', 'age15', 'age16', 'age17', 'age18', 'age19', 'age20',
    #                'age21', 'age22', 'age23', 'age24', 'age25', 'age26', 'age27', 'age28', 'age29', 'age30',
    #                'age31', 'age32', 'age33', 'age34', 'age35', 'age36', 'age37', 'age38', 'age39', 'age40',
    #                'age41', 'age42', 'age43', 'age44', 'age45', 'age46', 'age47', 'age48', 'age49', 'age50',
    #                'age51', 'age52', 'age53', 'age54', 'age55', 'age56', 'age57', 'age58', 'age59', 'age60',
    #                'age61', 'age62', 'age63', 'age64', 'age65', 'age66', 'age67', 'age68', 'age69', 'age70',
    #                'age71', 'age72', 'age73', 'age74', 'age75', 'age76', 'age77', 'age78', 'age79', 'age80',
    #                'age81', 'age82', 'age83', 'age84', 'age85', 'age86', 'age87', 'age88', 'age89', 'age90',
    #                'age91', 'age92', 'age93', 'age94', 'age95', 'age96', 'age97', 'age98', 'age99', 'age100',]  # 你的类别名称列表
    yaml_path = "C:\\Users\\HP\\Desktop\\Vision_Detect_homework\\bigwork\\task4\\dataset\\YOLO_dataset\\data.yaml"

    create_dataset_yaml(train_path, val_path, class_names, yaml_path)
