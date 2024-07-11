from __future__ import print_function, division

from torch.autograd import Variable
from torchvision.transforms import Normalize

import torch
from model.AerialNet import net_single_stream as net
from image.normalization import NormalizeImageDict, normalize_image
from util.checkboard import createCheckBoard
from geotnf.transformation import GeometricTnf, theta2homogeneous
from geotnf.point_tnf import *
from util.torch_util import print_info
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
import warnings

import pickle
from functools import partial

import time


warnings.filterwarnings('ignore')

# torch.cuda.set_device(1) # Using second GPU

### Parameter
feature_extraction_cnn = 'se_resnext101'
model_path = 'trained_models/checkpoint.pth.tar'

source_image_path='failure_cases/source_438.jpg'
target_image_path='failure_cases/target_438.jpg'

### Load models
use_cuda = torch.cuda.is_available()

# Create model
print('Creating CNN model...')
model = net(use_cuda=use_cuda, geometric_model='affine', feature_extraction_cnn=feature_extraction_cnn)
# net_single_stream - 不训练的时候使用Single Stream，训练的时候使用Double Stream

pickle.load = partial(pickle.load, encoding="latin1")   # 从已打开的 file object 文件 中读取封存后的对象，重建其中特定对象的层次结构并返回。它相当于 Unpickler(file).load()。
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1") # 它接受一个二进制文件用于读取 pickle 数据流。
# NOTE - 读取 NumPy array 和 Python 2 存储的 datetime、date 和 time 实例时，请使用 encoding='latin1'。
# <https://docs.python.org/zh-cn/3/library/pickle.html>

# Load trained weights
print('Loading trained model weights...')
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
# <https://blog.csdn.net/bc521bc/article/details/85623515> 将GPU上训练好的模型加载到CPU上
model.load_state_dict(checkpoint['state_dict'])
print("Reloading from--[%s]" % model_path)


### Load and preprocess images
resize = GeometricTnf(out_h=280, out_w=280, use_cuda=False)
normalizeTnf = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def Im2Tensor(image):
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    if use_cuda:
        image_var = image_var.cuda()
    return image_var

def preprocess_image(image: np.ndarray) -> torch.Tensor:    # 图像预处理
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)   # H,W,Channel -> Channel,H,W
    image = torch.Tensor(image.astype(np.float32) / 255.0, requires_grad=False)
    # image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image = resize(image)   # 这里没有输入仿射变换参数，只是对图像的大小做了一个变换
    # image_var = resize(image_var)

    # Normalize image
    image = normalize_image(image)
    # image_var = normalize_image(image_var)

    return image    # 返回重采样并归一化的图像tensor
    # return image_var

source_image = io.imread(source_image_path) # skimage.io.imread 返回的是 numpy.ndarray
target_image = io.imread(target_image_path)

source_image_var = preprocess_image(source_image)
target_image_var = preprocess_image(target_image)
target_image = np.float32(target_image/255.)

if use_cuda:
    source_image_var = source_image_var.cuda()
    target_image_var = target_image_var.cuda()

### Create image transformers
affTnf = GeometricTnf(geometric_model='affine', out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda=use_cuda)

batch = {'source_image': source_image_var, 'target_image':target_image_var}

# resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda = use_cuda)

### Evaluate model
model.eval()
# 可以选evaluate或者train，.eval()等价于.train(False)

start_time = time.time()
# Evaluate models
"""1st Affine"""
theta_aff, theta_aff_inv = model(batch)

# Calculate theta_aff_2
batch_size = theta_aff.size(0)
theta_aff_inv = theta_aff_inv.view(-1, 2, 3)    # 变成(batch_size,2,3)的仿射参数矩阵，加一行[0, 0, 1]就是单应矩阵
theta_aff_inv = torch.cat((theta_aff_inv, (torch.Tensor([0, 0, 1]).to('cuda').unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)
theta_aff_2 = theta_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

theta_aff_ensemble = (theta_aff + theta_aff_2) / 2  # Ensemble
# 这里用算数平均值

### Process result
warped_image_aff = affTnf(Im2Tensor(source_image), theta_aff_ensemble.view(-1,2,3))
result_aff_np = warped_image_aff.squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
io.imsave('results/aff.jpg', result_aff_np)

"""2nd Affine"""
# 给图像做两次warp，类似于dlk取两次地图模板
# Preprocess source_image_2
source_image_2 = normalize_image(resize(warped_image_aff.cpu()))
if use_cuda:
    source_image_2 = source_image_2.cuda()
theta_aff_aff, theta_aff_aff_inv = model({'source_image': source_image_2, 'target_image':batch['target_image']})

# Calculate theta_aff_2
batch_size = theta_aff_aff.size(0)
theta_aff_aff_inv = theta_aff_aff_inv.view(-1, 2, 3)
theta_aff_aff_inv = torch.cat((theta_aff_aff_inv, (torch.Tensor([0, 0, 1]).to('cuda').unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)
theta_aff_aff_2 = theta_aff_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

theta_aff_aff_ensemble = (theta_aff_aff + theta_aff_aff_2) / 2  # Ensemble

theta_aff_ensemble = theta2homogeneous(theta_aff_ensemble)
theta_aff_aff_ensemble = theta2homogeneous(theta_aff_aff_ensemble)

theta = torch.bmm(theta_aff_aff_ensemble, theta_aff_ensemble).view(-1, 9)[:, :6]

### Process result
warped_image_aff_aff = affTnf(Im2Tensor(source_image), theta.view(-1,2,3))
result_aff_aff_np = warped_image_aff_aff.squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
io.imsave('results/aff_aff.jpg', result_aff_aff_np)

print()
print_info("# ====================================== #\n"
           "#            <Execution Time>            #\n"
           "#            - %.4s seconds -            #"%(time.time() - start_time)+"\n"
           "# ====================================== #",['yellow','bold'])

# Create overlay
aff_overlay = cv2.addWeighted(src1=result_aff_np, alpha= 0.4, src2=target_image, beta=0.8, gamma=0)
io.imsave('results/aff_overlay.jpg', np.clip(aff_overlay,-1,1))

# Create checkboard
aff_checkboard = createCheckBoard(result_aff_np, target_image)
io.imsave('results/aff_checkboard.jpg', aff_checkboard)

### Display
fig, axs = plt.subplots(2,3)
axs[0][0].imshow(source_image)
axs[0][0].set_title('Source')
axs[0][1].imshow(target_image)
axs[0][1].set_title('Target')
axs[0][2].imshow(result_aff_np)
axs[0][2].set_title('Affine')

axs[1][0].imshow(result_aff_aff_np)
axs[1][0].set_title('Affine X 2')
axs[1][1].imshow(aff_checkboard)
axs[1][1].set_title('Affine Checkboard')
axs[1][2].imshow(aff_overlay)
axs[1][2].set_title('Affine Overlay')

for i in range(2):
    for j in range(3):
        axs[i][j].axis('off')
fig.set_dpi(300)
plt.show()

