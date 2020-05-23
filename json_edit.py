import glob
import os
import json

path = "C:\\Users\\liangyi\\Desktop\\wxpython_camera\\"

file_format = "*.json"
file_list = glob.glob(path + file_format)

use_method = "replace"

if use_method == "replace":
    obj = "mouse"
    replace = "mouth"
    for i in iter(file_list):
        is_exists = os.path.exists(i)
        if is_exists:
            # 打开json文件
            f = open(i, encoding='utf-8')
            # 读取json
            setting = json.load(f)
            for index in range(len(setting['shapes'])):
                if setting['shapes'][index]['label'] == "pp_wall":
                    setting['shapes'][index]['label'] = "pp-wall"
            # setting['imagePath'] = setting['imagePath'].replace("../image/rgb/","")
            string = json.dumps(setting)
            with open(i, 'w', encoding='utf-8') as f:
                f.write(string)
                f.close()
            print(i+",修复完成")
        else:
            continue

# 通过.json连jpg也改了

elif use_method == "rename":
    obj = ".json"
    replace = ".jpg"
    suffix = "_zed"
    for i in iter(file_list):
        is_exists = os.path.exists(i)
        if is_exists:
            jpg_file = i.replace(obj, replace)
            jpg_file = jpg_file.replace("\\label\\", "\\image\\")
            rename_file = jpg_file.replace(replace, suffix+replace)
            rename_file = rename_file.replace("\\label\\", "\\image\\")
            # 图片rename
            os.rename(jpg_file, rename_file)
            f = open(i, encoding='utf-8')
            # 读取json
            setting = json.load(f)
            setting['imagePath'] = setting['imagePath'].replace(replace, suffix+replace)
            string = json.dumps(setting)
            with open(i, 'w', encoding='utf-8') as f:
                f.write(string)
                f.close()
            os.rename(i, i.replace(obj, suffix+obj))
            print(rename_file)
        else:
            continue
