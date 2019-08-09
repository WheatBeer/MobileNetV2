import os
import time
import copy
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from utils.mobilenetv2 import MobileNetV2
from utils import accuracy, AverageMeter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Training MobileNetV2')

parser.add_argument('-b', '--batch', type=int, metavar='#', 
                    default=400, help='Batch Size | Default: 400')

parser.add_argument('-e', '--epoch', type=int, metavar='#',
                    default=1, help='Epoches | Default: 1')

parser.add_argument('-l', '--lr', type=float, metavar='#', 
                    default=0.045, help='Learning Rate | Default: 0.045')

parser.add_argument('-n', '--n-digits', type=int, metavar='#', 
                    default=0, help='Round N digits | Default: 0')

parser.add_argument('-d', '--decay', type=bool, metavar='T/F',
                    default=True, help='Learning Rate Decay | Default: True')

parser.add_argument('--pretrained', type=bool, metavar='T/F', 
                    default=True, help='Train from pretrained model | Default: True')

parser.add_argument('-s', '--save', type=bool, metavar='T/F', 
                    default=True, help='Save all models after every epoch(True) | Save best model(False)')

parser.add_argument('-p', '--path', metavar='PATH', 
                    default='/Data/ImageNet/ILSVRC2012/', 
                    help='Imagenet Dataset PATH | Default: /Data/ImageNet/ILSVRC2012/')

parser.add_argument('-w', '--weights', metavar='PATH', 
                    default='./data/mobilenet_v2-b0353104.pth',
                    help='Pretrained parameters PATH | Default: ./data/mobilenet_v2-b0353104.pth')

def round_tensor(arr, n_digits):
    rounded = torch.round(arr * 2**n_digits) / (2**n_digits)
    return rounded

def train(train_loader, model, criterion, optimizer, decay):
    # Initialize values
    batch_time = AverageMeter('Batch',':.3f')
    data_time = AverageMeter('Data',':.3f')
    losses = AverageMeter('Loss',':.4f')
    top1 = AverageMeter('Top1',':.4f')
    top5 = AverageMeter('Top5',':.4f')
    
    # switch to train mode
    model.train()

    end = time.time()
    for i, (inputs, target) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, target = inputs.cuda(), target.cuda()

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if decay: 
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Print results
    print('===> Total: {total:2.2f}m | Loss: {loss:2.2f} | top1: {top1: 2.3f} | top5: {top5: 2.3f}'.format(
        total=(batch_time.sum+data_time.sum)/60,
        loss=losses.avg,
        top1=top1.avg,
        top5=top5.avg,))

def validate(test_loader, model, criterion):
    # Initialize values
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Top 1 accuracy', ':6.3f')
    top5 = AverageMeter('Top 5 accuracy', ':6.3f')
    
    # switch to validate mode
    model.eval()
    
    # Foward
    print('===> Start validation!')
    with torch.no_grad():
        for i, (inputs, target) in enumerate(tqdm(test_loader)):
            
            inputs, target = inputs.cuda(), target.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            prec1, prec5 = accuracy(outputs.data, target.data, topk=(1, 5))
            losses.update(loss.data, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
    print('===>', top1.__str__(),'\t', top5.__str__())
    print('===> Validation is done!')
    return (losses.avg, top1.avg)
    
def main():
    args = parser.parse_args()
    epoches = args.epoch
    best_prec1 = 0
    multi_gpu = False
    
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagenet_path = args.path
    train_dir = os.path.join(imagenet_path, 'train')
    train_set = datasets.ImageFolder(train_dir, preprocess)
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=args.batch,
                                              shuffle=True,
                                              num_workers=8)
    test_dir = os.path.join(imagenet_path, 'val')
    test_set = datasets.ImageFolder(test_dir, preprocess)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch,
                                              shuffle=True,
                                              num_workers=8)
    
    print('===> Image preprocessing is done!')
    print('===> Batch size:', args.batch) 
    
    # Define the model
    model = MobileNetV2()
    if args.pretrained:
        model.load_state_dict(torch.load(args.weights))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
      print('===> You have to use GPUs')
      exit()
    print('===> Test on', device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        multi_gpu = True
        model = nn.DataParallel(model)
        print('===> Using', torch.cuda.device_count(), 'GPUs!')
    
    # round up
    if args.n_digits > 0:
        print('===> Round up(', args.n_digits,')')
        for param in model.parameters():
            param.data = round_tensor(param.data, args.n_digits)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    # learning rate decay rate of 0.98 per epoch
    if args.decay:
        scheduler = StepLR(optimizer, step_size=1, gamma=0.98)
    
    for epoch in range(epoches):
        print('===> epoch:',epoch+1)
        # train for one epoch
        train(train_loader, model, criterion, optimizer, args.decay)
        
        # evaluate on validation set
        val_loss, prec1 = validate(test_loader, model, criterion)
        
        if args.save: # save all models after every epoch
            epoch_s = str(epoch)
            save_file_name = './data/retrain_data' + epoch_s + '.pth'
            if multi_gpu:
                torch.save(model.module.state_dict(), save_file_name)
            else:
                torch.save(model.state_dict(), save_file_name)
        else: # save best model
            if prec1 > best_prec1:
                best_prec1 = prec1
                best_model = copy.deepcopy(model)
                if multi_gpu:
                    torch.save(best_model.module.state_dict(),'./data/best_model.pth')
                else:
                    torch.save(best_model.state_dict(),'./data/best_model.pth')

if __name__ == '__main__':
    main()
