import torch
from PIL import Image
import os
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable
import time
import numpy as np
# image_folder = './test_images/'

save_folder = './save_model5_lap/'
# image_list = os.listdir(image_folder)
image_list = []
fp = open('list','r')
for line in fp.readlines():
     line = line.strip()
     image_list.append(line)
model = torch.load('./model_adam/model_epoch_5.pth')["model"]

model = model.cpu()
count = 0
num = 0
for image_name in image_list:
     # image = Image.open(os.path.join(image_folder, image_name))
     # print(os.path.join(image_folder, image_name))
     image = Image.open(image_name).convert('YCbCr')
     y,cb,cr = image.split()
     input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
     # input = input.cuda()
     start = time.time()
     out = model(input)
     out = out.cpu()
     out_y = out.data[0].numpy()
     out_y *= 255.0
     out_y = out_y.clip(0,255)
     out_y = Image.fromarray(np.uint8(out_y[0]), mode='L')
     out_image_cb = cb.resize(out_y.size,Image.BICUBIC)
     out_image_cr = cr.resize(out_y.size,Image.BICUBIC)
     out_img = Image.merge('YCbCr', [out_y, out_image_cb, out_image_cr]).convert('RGB')
     save_image_name = image_name.split('/')[-1]
     out_img.save(os.path.join(save_folder, save_image_name))
     end = time.time()
     count += end - start
     num += 1
     print "num:", num
     #save_image_name = image_name.split('/')[-1]
     #out_image.save(os.path.join(save_folder, save_image_name))
print "time: ", count
