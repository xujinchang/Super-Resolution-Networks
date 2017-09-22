import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os
from torchvision.transforms import ToTensor, ToPILImage
from torch.autograd import Variable
 
image_folder = './test_images/'
 
save_folder = './save/'
 
image_list = os.listdir(image_folder)
 
model = torch.load('./model/model_epoch_8.pth')["model"]

model = model.cpu()
for image_name in image_list:
     image = Image.open(os.path.join(image_folder, image_name))
     print(os.path.join(image_folder, image_name))
     input = Variable(ToTensor()(image)).view(1, -1, image.size[1], image.size[0])
     # input = input.cuda()
     out = model(input)
     out = out.cpu()
     out = out.data[0]
     print out
     out[out > 1] = 1 
     out[out < 0] = 0 
     out_image = ToPILImage()(out)
     out_image.save(os.path.join(save_folder, image_name))
