# -*-coding: utf-8 -*-
"""
    @Author : Pan
    @E-mail : 390737991@qq.com
    @Date   : 2022-08-25 18:43:16
    @Brief  :
"""

import os
import random
from tqdm import tqdm
from pybaseutils import file_utils, image_utils

if __name__ == "__main__":
    image_dir = "/home/data/nasdata/dataset/csdn/animal/animals10/train"
    filename = os.path.join(os.path.dirname(image_dir), "class_name.txt")
    labels = file_utils.get_sub_paths(image_dir)
    file_utils.write_list_data(filename, labels)
