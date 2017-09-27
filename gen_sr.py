import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable
import time

# image_folder = './test_images/'
 
save_folder = './save_model3_vgg/'
 
# image_list = os.listdir(image_folder)
image_list = []
fp = open('list','r')
for line in fp.readlines():
     line = line.strip()
     image_list.append(line)
model = torch.load('./model_2_vgg/model_epoch_3.pth')["model"]

model = model.cpu()
count = 0
for image_name in image_list:
     # image = Image.open(os.path.join(image_folder, image_name))
     # print(os.path.join(image_folder, image_name))
     image = Image.open(image_name)
     input = Variable(ToTensor()(image)).view(1, -1, image.size[1], image.size[0])
     # input = input.cuda()
     start = time.time()
     out = model(input)
     out = out.cpu()
     out = out.data[0]
     out[out > 1] = 1 
     out[out < 0] = 0 
     out_image = ToPILImage()(out)
     end = time.time()
     count += end - start
     save_image_name = image_name.split('/')[-1]
     out_image.save(os.path.join(save_folder, save_image_name))
print "time: ", count