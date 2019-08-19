import os
import argparse
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.datasets as datasets
from utils.mobilenetv2 import MobileNetV2
from utils import accuracy, AverageMeter
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Float Shift Validation')

parser.add_argument('-b', '--batch', type=int, metavar='#', 
                    default=400, help='Batch Size | Default: 400')

parser.add_argument('-p', '--path', metavar='PATH', 
                    default='/Data/ImageNet/ILSVRC2012/', 
                    help='Imagenet Dataset PATH | Default: /Data/ImageNet/ILSVRC2012/')

parser.add_argument('-w', '--weights', metavar='PATH', 
                    default='./data/mobilenet_v2-b0353104.pth',
                    help='Pretrained parameters PATH | Default: ./data/mobilenet_v2-b0353104.pth')

def main():
    args = parser.parse_args()
    
    print('===> Start image preprocessing!')
    # Data Load & Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    imagenet_path = args.path
    test_dir = os.path.join(imagenet_path, 'val')

    test_set = datasets.ImageFolder(test_dir, preprocess)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch,
                                              shuffle=True,
                                              num_workers=4)
    
    print('===> 1000 classes x 50 images preprocessing is done!')
    print('===> batch size:', args.batch) 
    
    # Initialize values
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Top 1 accuracy', ':6.3f')
    top5 = AverageMeter('Top 5 accuracy', ':6.3f')
    
    # Define the model
    model = MobileNetV2()
    criterion = nn.CrossEntropyLoss()
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('===> Test on', device)
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('===> Using', torch.cuda.device_count(), 'GPUs!')

    # Foward
    print('===> Start inferencing!')
    with torch.no_grad():
        for i, (inputs, target) in enumerate(tqdm(test_loader)):
            
            inputs, target = inputs.to(device), target.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, target)
            
            prec1, prec5 = accuracy(outputs.data, target.data, topk=(1, 5))
            losses.update(loss.data, inputs.size(0))
            top1.update(prec1, inputs.size(0))
            top5.update(prec5, inputs.size(0))
        print('===>', top1.__str__(),'\t', top5.__str__())
    print('===> Test is done!')
    return (losses.avg, top1.avg)


if __name__ == '__main__':
    main()
