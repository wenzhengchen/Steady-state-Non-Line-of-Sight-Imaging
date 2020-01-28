

import argparse


def get_args():
    parser = argparse.ArgumentParser(description='nlos')
    
    parser.add_argument('--datafolder', default='/u6/a/wenzheng/remote3/datasets/shapenet/mnist-render-steady', type=str, help='data folder')
    parser.add_argument('--shininess', type=int, default=0, help='shininess level')
    parser.add_argument('--catagory', default='pillow', type=str, help='shapenet class')
    parser.add_argument('--svfolder', default='./model', type=str, help='result folder')
    
    parser.add_argument('--gf_dim', type=int, default=128,
                        help='dim of channel')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='dim of output, gray')
    parser.add_argument('--in_dim', type=int, default=3,
                        help='dim of input, gray+xy')
    
    parser.add_argument('--mergelow', type=float, default=0.1,
                        help='mergelow')
    parser.add_argument('--mergehigh', type=float, default=0.2,
                        help='mergehigh')
    
    args = parser.parse_args()

    return args

