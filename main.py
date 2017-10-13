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
from resnet7 import Net
from data import get_training_set, get_test_set
from Gan import discriminator
#training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--train_dir', type=str, help='train_dataset directory path')
parser.add_argument('--label_dir', type=str, help='label_dataset directory path')
parser.add_argument('--upscale_factor', type=int, required=True, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument('--lrD', type=float, default=0.01, help='Learning Rate. Default=0.01')
parser.add_argument("--step", type=int, default=3, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=500")
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, help="weight decay, Default: 0")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained model (default: none)")
parser.add_argument("--vgg_loss", action="store_true", help="Use content loss?")
parser.add_argument("--Gan_loss", action="store_true", help="Use Gan loss?")
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
# train_set = get_training_set(opt.train_dir, opt.label_dir)
# test_set = get_test_set(opt.train_dir, opt.label_dir)
#train_set = get_training_set('/localSSD/zhaoyu/super_resolution/DIV2K_train_LR_bicubic/X2', '/localSSD/zhaoyu/super_resolution/DIV2K_train_HR')
#test_set = get_test_set('/localSSD/zhaoyu/super_resolution/DIV2K_train_LR_bicubic/X2', '/localSSD/zhaoyu/super_resolution/DIV2K_train_HR')
# training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
# testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
# print('===> Building model')

# train_set = get_training_set('/home/tmp_data_dir/zhaoyu/CelebA/img_align_celeba/', '/home/tmp_data_dir/zhaoyu/CelebA/img_align_celeba/')

train_set = get_training_set('/home/xujinchang/pytorch-CycleGAN-and-pix2pix/datasets/celeA_part/train/', '/home/xujinchang/pytorch-CycleGAN-and-pix2pix/datasets/celeA_part/train/')
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)


if opt.vgg_loss:
    print('===> Loading VGG model')
    netVGG = models.vgg19()
    netVGG.load_state_dict(model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'))
    # netVGG = torch.load('/home/xujinchang/.torch/models/vgg19-dcbb9e9d.pth')
    # print checkpoint

    # netVGG.load_state_dict('/home/xujinchang/.torch/models/vgg19-dcbb9e9d.pth')
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
print model
criterion_mse = nn.MSELoss()
device_id = 0
if cuda:
    model = model.cuda(device_id)
    criterion_mse = criterion_mse.cuda(device_id)
    if opt.vgg_loss:
        netContent = netContent.cuda(device_id)

if opt.Gan_loss:
    print('====> Using Gan Loss')
    D_network = discriminator()
    print(D_network)
   
    criterion_gan = nn.BCELoss()
    if cuda:
        D_network = D_network.cuda(device_id)
        criterion_gan = criterion_gan.cuda(device_id)


    # D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

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
if opt.Gan_loss:
    D_optimizer = optim.Adam(D_network.parameters(), lr=opt.lrD)


def adjust_learning_rate_SR(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr


def adjust_learning_rate_GAN(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10"""
    lrD = opt.lrD * (0.1 ** (epoch // opt.step))
    return lrD

def train(training_data_loader, optimizer_G, optimizer_D, G_network, D_network, VGG_Content, criterion_MSE, criterion_GAN, epoch):
    lr = adjust_learning_rate_SR(optimizer_G, epoch-1)
    for param_group in optimizer_G.param_groups:
        param_group["lr"] = lr

    lrD = adjust_learning_rate_GAN(optimizer_D, epoch-1)
    for param_group in optimizer_D.param_groups:
        param_group["lr"] = lrD
    print "epoch =", epoch, "lr =", optimizer_G.param_groups[0]["lr"]
    print "epoch =", epoch, "lrD =", optimizer_D.param_groups[0]["lr"]

    G_network.train()

    D_network.train()


    for iteration, batch in enumerate(training_data_loader, 1):
        LR, target = Variable(batch[0]), Variable(batch[1])
        y_real_, y_fake_ = Variable(torch.ones(int(LR.size()[0]), 1)), Variable(torch.zeros(int(LR.size()[0]), 1))

        if cuda:
            LR = LR.cuda(device_id)
            target = target.cuda(device_id)
            y_real_ = y_real_.cuda(device_id)
            y_fake_ = y_fake_.cuda(device_id)

        out = G_network(LR)

        if opt.Gan_loss:

            optimizer_D.zero_grad()
            x_fake_ = D_network(out)
            x_real_ = D_network(target)

            D_fake_loss = criterion_GAN(x_fake_, y_fake_)
            D_real_loss = criterion_GAN(x_real_, y_real_)
            
            D_loss = D_fake_loss + D_real_loss
            D_loss.backward(retain_graph=True)
            optimizer_D.step()

        optimizer_G.zero_grad()
        
        MSE_loss = criterion_MSE(out, target)
        MSE_loss.backward(retain_graph=True)


        if opt.Gan_loss:
            x_fake_ = D_network(out)
            G_loss = criterion_GAN(x_fake_, y_real_)
            G_loss.backward(retain_graph=True)

        if opt.vgg_loss:
            content_input = VGG_Content(out)
            content_target = VGG_Content(target)
            content_target = content_target.detach()
            VGG_loss = criterion_MSE(content_input, content_target)
            VGG_Content.zero_grad()
            VGG_loss.backward(retain_graph=True)

        
        optimizer_G.step()

        if iteration%100 == 0:
            if opt.vgg_loss:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f} Content_loss {:.10f} G_loss {:.10f} D_loss {:.10f}".format(epoch, iteration, len(training_data_loader), MSE_loss.data[0], VGG_loss.data[0], G_loss.data[0], D_loss.data[0]))
            else:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), MSE_loss.data[0]))

def save_checkpoint(model, epoch):
    model_out_path = "model_2_vgg_gan/" + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("model_2_vgg_gan/"):
        os.makedirs("model_2_vgg_gan/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))



for epoch in range(1, opt.nEpochs + 1):
        train(training_data_loader, optimizer, D_optimizer, model, D_network, netContent, criterion_mse, criterion_gan, epoch)
        save_checkpoint(model, epoch)


