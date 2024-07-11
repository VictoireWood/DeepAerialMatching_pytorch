'''
有个致命的问题，由于randint的范围一直是整个能取的范围，所以不确定会不会有两轮循环取出来的两组图片位置距离过近，实际上可以合并。
有个办法就是每取出一组图片就把可能重复的范围从原来的范围中挖去，把剩余可取的范围做成一个列表，下次randint的时候在列表中随机取一个区间进行randint。
这个功能比较麻烦，最后再加。

不行，取不满足够的图片，应该去掉的是x和y横竖两条的并集而不是交集。
记录所有随机取到的标准中心点，用最近邻搜索来删掉后来取得太近的。
如果太近就重新取，保证能取满1000个。
'''

import os
import sys
import shutil
from glob import glob

import random
from tqdm import tqdm
import pandas as pd
import math
from os.path import join
import numpy as np
from itertools import product
import cv2
import json
from pyproj import Transformer
from haversine import haversine, Unit
from sklearn.neighbors import NearestNeighbors

import torch
from math import cos, sin, pi
import re
from itertools import combinations

# size scale range
min_scale = 0.75
max_scale = 1.25

# rotation range (-angle_range, angle_range)
angle_range = 15  # degrees

# projective variables (p7, p8)
projective_range = 0

# translation (p3, p6)
translation_range = 10  # pixels

CITIES = [
    'CITY_1',
    'CITY_2',
    'CITY_3',
    'CITY_4',
    'CITY_5',
    'CITY_6',
    'CITY_7'
]

def create_noexist_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

Match_Dataset_name = 'Cities'
CSV_DIR = f'./training_data/{Match_Dataset_name}/'
BASE_DIR = r'E:\GeoVINS\AerialVL\vpr_training_data'
map_database_path = BASE_DIR + r'\raw_satellite_imageries\*'
# origin_map_tile_path = BASE_DIR + r'\before_affine' + '\\'
origin_map_tile_path = f'../{Match_Dataset_name}/origin/'
map_tile_pairs_path = f'../{Match_Dataset_name}/paired/'
create_noexist_dir(CSV_DIR)
create_noexist_dir(origin_map_tile_path)
create_noexist_dir(map_tile_pairs_path)

def random_theta_generator():
    # create random ground truth warp parameters in the specified ranges

    scale = random.uniform(min_scale, max_scale)
    angle = random.uniform(-angle_range, angle_range)
    translation_x = random.uniform(-translation_range, translation_range)
    translation_y = random.uniform(-translation_range, translation_range)

    rad_ang = angle / 180 * pi
    USE_CUDA = torch.cuda.is_available()

    if USE_CUDA:
        theta = torch.Tensor([scale * cos(rad_ang),
                            -scale * sin(rad_ang),
                            translation_x,
                            scale * sin(rad_ang),
                            scale * cos(rad_ang),
                            translation_y]).cuda()
    else:
        theta = torch.Tensor([scale * cos(rad_ang),
                            -scale * sin(rad_ang),
                            translation_x,
                            scale * sin(rad_ang),
                            scale * cos(rad_ang),
                            translation_y])

    # theta = theta.unsqueeze(0)
    theta = theta.view(2, 3).numpy()

    return theta

def generate_match_dataset():
    # important parameters
    map_w = 5000    # TODO - 原始图像的像素宽
    map_h = 5000

    img_w = 800     # TODO - 目标图像所在区域在原始图像上所占的像素宽和高
    img_h = 800

    target_w = 500  # TODO - 目标图像的像素宽（需要resize）
    target_h = 500

    shift_range = 100    # TODO - 区别两个地点的间距的最小距离

    train_set = 1000        # 这个是每个城市的每对map生成训练集数量的下限
    train_val_ratio = 0.3
    val_set = int(train_set * train_val_ratio)
    set_volume = [train_set, val_set]

    header = pd.DataFrame(columns=['source_path', 'target_path', 'city_id', 'origin_img', 'pixel_loc_x', 'pixel_loc_y', 'A11', 'A12', 'tx', 'A21', 'A22', 'ty'])
    csv_path_train = CSV_DIR + 'train_pairs.csv'
    csv_path_val = CSV_DIR + 'val_pairs.csv'
    csv_paths = [csv_path_train, csv_path_val]
    for csv_idx in range(2):
        header.to_csv(csv_paths[csv_idx], mode='w', index=False, header=True)

    # NOTE - csv，记录图像对的路径、仿射变换参数、地图的经纬度范围、city
    cities_path = glob(map_database_path)
    for city_path in cities_path:
        city_id = city_path.split('\\')[-1]
        exclude_substitute = re.compile('^(?!.*_sub)')
        map_list_full = glob(os.path.join(city_path, '*.png'))
        map_list = []

        # 去掉之前生成的重复的图像
        for map_path in map_list_full:
            res = re.search(exclude_substitute, map_path)
            if res:
                map_list.append(map_path)
            else:
                pass
            del res
        
        for check_idx in map_list:
            check_map = cv2.imread(check_idx)
            if check_map.shape[0] != map_w or check_map.shape[1] != map_h:
                sys.exit('Wrong map shape!')
            else:
                pass
        del check_idx, check_map

        map_combinations = list(combinations(map_list, 2))
        combinations_num = len(map_combinations)
        

        taken_points = []   # 已经取过的点

        for map_pair in map_combinations:
            map_src_origin_path = map_pair[0]
            map_src_origin = map_src_origin_path.split('\\')[-1].replace('.png', '')
            map_trg_origin_path = map_pair[1]
            map_trg_origin = map_trg_origin_path.split('\\')[-1].replace('.png', '')
            for set_idx in range(2):
                for generate_loop in range(set_volume[set_idx]):
                    # 确保随机取的像素点位置和已经取过的所有点至少隔了shift_range个像素的长度
                    while True:
                        loc_x = random.randint(0, map_w - img_w)
                        loc_y = random.randint(0, map_h - img_h)
                        if len(taken_points) == 0:
                            break
                        else:
                            neigh = NearestNeighbors(n_neighbors=1, radius=shift_range)
                            neigh.fit(taken_points)
                            idx = neigh.radius_neighbors([[loc_x, loc_y]], return_distance=False)
                            number = idx.item().size
                            if number == 0:
                                break
                    taken_points.append([loc_x, loc_y])
                    for map_path in map_list:
                        map_origin = cv2.imread(map_path)
                        map_slice = map_origin[loc_y:loc_y + img_h, loc_x:loc_x + img_w]
                        map_resize = cv2.resize(map_slice, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        origin_img = map_path.split('\\')[-1].replace('.png', '')
                        img_name = f'@{city_id}@{origin_img}@{loc_x}@{loc_y}@.png'
                        cv2.imwrite(origin_map_tile_path + img_name, map_resize)
                    
                    
                    img_src_name = f'@{city_id}@{map_src_origin}@{loc_x}@{loc_y}@.png'
                    img_src_path = origin_map_tile_path + img_src_name
                    img_trg_name = f'@{city_id}@{map_trg_origin}@{loc_x}@{loc_y}@.png'
                    img_trg_path = origin_map_tile_path + img_trg_name

                    theta = random_theta_generator()

                    data_line = pd.DataFrame([[img_src_path, img_trg_path, city_id, origin_img, loc_x, loc_y, theta[0,0], theta[0,1], theta[0,2], theta[1,0], theta[1,1], theta[1,2]]], columns=['source_path', 'target_path', 'city_id', 'origin_img', 'pixel_loc_x', 'pixel_loc_y', 'A11', 'A12', 'tx', 'A21', 'A22', 'ty'])
                    data_line.to_csv(csv_paths[set_idx], mode='a', index=False, header=False)



if __name__ == '__main__':
    generate_match_dataset()
    pass