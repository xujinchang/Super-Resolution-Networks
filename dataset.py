import torch.utils.data as data
from torchvision.transforms import Compose, CenterCrop, ToTensor, Scale

from os import listdir
from os.path import join
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".JPEG"])


def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath).convert('YCbCr')
    # y, cb, cr = img.split()
    # return y
    return img

def read_image(directory):
    image_filename = [join(directory,x) for x in listdir(directory)]
    return image_filename
def read_image_label(directory):
    filename = sorted(listdir(directory))
    #print(filename[-1])
    #print(listdir(directory+filename[-1]))
    for file in filename[-1:]:
        image_filename = [join(directory+file,x) for x in listdir(directory+file)]
    #print(sorted(image_filename))
    return sorted(image_filename)



class DatasetFromFolder1(data.Dataset):
    def __init__(self, train_dir, label_dir, input_transform=None, target_transform1=None,target_transform2=None):
        super(DatasetFromFolder1, self).__init__()
        self.train_filename = read_image(train_dir)
        self.label_filename = self.train_filename
        self.input_transform = input_transform
        self.target_transform1 = target_transform1
        self.target_transform2 = target_transform2
    def __getitem__(self, index):
        input = load_img(self.train_filename[index])
        target = load_img(self.label_filename[index])
        if self.target_transform1:
            target1 = self.target_transform1(target)
        if self.target_transform2:
            target2 = self.target_transform2(target)
        if self.input_transform:
            input = self.input_transform(input) #downsample
        return input, target1,target2

    def __len__(self):
        print (len(self.train_filename))
        print (len(self.label_filename))
        return len(self.train_filename)


class DatasetFromFolder2(data.Dataset):
    def __init__(self, train_dir, label_dir, input_transform=None, target_transform1=None):
        super(DatasetFromFolder2, self).__init__()
        self.train_filename = read_image(train_dir)
        self.label_filename = self.train_filename
        self.input_transform = input_transform
        self.target_transform1 = target_transform1
       
    def __getitem__(self, index):
        input = load_img(self.train_filename[index])
        target = load_img(self.label_filename[index])
        if self.target_transform1:
            target1 = self.target_transform1(target)

        if self.input_transform:
            input = self.input_transform(input) #downsample
        return input, target1

    def __len__(self):
        print (len(self.train_filename))
        print (len(self.label_filename))
        return len(self.train_filename)
