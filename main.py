from __future__ import print_function
import argparse, os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import models
import torch.utils.model_zoo as model_zoo
from srresnet import Net


#training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--train_dir', type=str, help='train_dataset directory path')
parser.add_argument('--label_dir', type=str, help='label_dataset directory path')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument("--step", type=int, default=500, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set(opt.train_dir, opt.label_dir)
test_set = get_test_set(opt.train_dir, opt.label_dir)
#train_set = get_training_set('/localSSD/zhaoyu/super_resolution/DIV2K_train_LR_bicubic/X2', '/localSSD/zhaoyu/super_resolution/DIV2K_train_HR')
#test_set = get_test_set('/localSSD/zhaoyu/super_resolution/DIV2K_train_LR_bicubic/X2', '/localSSD/zhaoyu/super_resolution/DIV2K_train_HR')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
# print('===> Building model')

if opt.vgg_loss:
	print('===> Loading VGG model')
    netVGG = models.vgg19()
    netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
    class _content_model(nn.Module):
        def __init__(self):
            super(_content_model, self).__init__()
            self.feature = nn.Sequential(*list(netVGG.features.children())[:-1])
                
        def forward(self, x):
            out = self.feature(x)
            return out

    netContent = _content_model()

print("===> Building model")

model = Net(opt.upscale_factor)
criterion1 = nn.MSELoss()
device_id = 1

if cuda:
    model = model.cuda(device_id)
    criterion1 = criterion1.cuda(device_id)
    if opt.vgg_loss:
            netContent = netContent.cuda() 

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
            
    # optionally copy weights from a checkpoint
if opt.pretrained:
	if os.path.isfile(opt.pretrained):
		print("=> loading model '{}'".format(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights['model'].state_dict())
    else:
        print("=> no model found at '{}'".format(opt.pretrained))

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr 

def train(training_data_loader, optimizer, model, criterion, epoch):
	lr = adjust_learning_rate(optimizer, epoch-1)
    
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr  

    print "epoch =", epoch,"lr =",optimizer.param_groups[0]["lr"]
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])

        if cuda:
            input = input.cuda(device_id)
            target = target.cuda(device_id)

        optimizer.zero_grad()
        out = model(input)
        loss = criterion(out, target)
       
        if opt.vgg_loss:
        	content_input = netContent(output)
            content_target = netContent(target)
            content_target = content_target.detach()
            content_loss = criterion(content_input, content_target)
        
        optimizer.zero_grad()

        if opt.vgg_loss:
        	netContent.zero_grad()
            content_loss.backward(retain_variables=True)
        
        loss.backward()
        optimizer.step()

        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} Content_loss {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0], content_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "model/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model/"):
        os.makedirs("model/")

    torch.save(state, model_out_path)
        
    print("Checkpoint saved to {}".format(model_out_path))



for epoch in range(1, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)


