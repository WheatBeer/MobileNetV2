import argparse
import struct
import torch
import torch.nn as nn
from utils.mobilenetv2 import MobileNetV2
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Mask Weights(FP32) Fractional')

parser.add_argument('-n', '--n-digits', type=int, metavar='#',
                    default=5, help='Fractional bits | Default: 5')

parser.add_argument('-w', '--weights', metavar='PATH',
                    default='./data/mobilenet_v2-b0353104.pth',
                    help='Pretrained parameters PATH | Default: ./data/mobilenet_v2-b0353104.pth')

# def masked_fp32(num, n_digits):
#     string_bin = format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')
#     masked = string_bin[:9+n_digits] + '0'*(23-n_digits)
#     return struct.unpack('!f',struct.pack('!I', int(masked, 2)))[0]

def masked_fp32(num, n_digits):
    string_bin = format(struct.unpack('!I', struct.pack('!f', num))[0], '032b')
    exp = struct.unpack('i', struct.pack('i', int(string_bin[1:9], 2)))[0]
    if exp <= 112: # 3bits: 123 | 4bits 112
        return 0
    else:
        masked = string_bin[:9+n_digits] + '0'*(23-n_digits)
        return struct.unpack('!f',struct.pack('!I', int(masked, 2)))[0]

def main():
    args = parser.parse_args()

    model = MobileNetV2()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    for i, param in enumerate(tqdm(model.parameters())):
        if len(param.shape) == 1:
            for w in range(param.shape[0]):
                param[w] = masked_fp32(param[w], args.n_digits)
        if len(param.shape) == 2:
            for w in range(param.shape[0]):
                for x in range(param.shape[1]):
                    param[w][x] = masked_fp32(param[w][x], args.n_digits)
        if len(param.shape) == 4:
            for w in range(param.shape[0]):
                for x in range(param.shape[1]):
                    for y in range(param.shape[2]):
                        for z in range(param.shape[3]):
                            param[w][x][y][z] = masked_fp32(param[w][x][y][z], args.n_digits)

    save_file_name = './data/masked_exp' + str(args.n_digits) + '.pth'
    torch.save(model.state_dict(), save_file_name)

if __name__ == '__main__':
    main()