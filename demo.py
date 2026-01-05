import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from configs.default import get_cfg
from configs.things_eval import get_cfg as get_things_cfg
from configs.small_things_eval import get_cfg as get_small_things_cfg
from core.utils.misc import process_cfg
import datasets
from utils import flow_viz
from utils import frame_utils
from utils.flow_viz import write_flo, flow2rgb
# from FlowFormer import FlowFormer
from core.FlowFormer import build_flowformer
from raft import RAFT

from utils.utils import InputPadder, forward_interpolate

from imageio import imwrite
from scipy import misc, io

DEVICE = 'cuda'
def load_image(imfile):
    # img = np.array(Image.open(imfile).convert('RGB')).astype(np.uint8)
    img = np.array(Image.open(imfile).resize((720,480))).astype(np.uint8)
    # img = np.array(Image.open(imfile)).astype(np.uint8)

    # 8 bit->24 bit
    # img = np.stack((img,) * 3, axis=-1)

    # img = np.resize(img, (540, 960, 3))
    print(img.shape)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


flow_save_path = './result/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--img_path', help="path to the image file")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    # cfg = get_cfg()
    if args.small:
        cfg = get_small_things_cfg()
    else:
        cfg = get_things_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_flowformer(cfg))
    model.load_state_dict(torch.load(args.model))

    print(args)

    model.cuda()
    model.eval()

    for idx in range(100):
        img_id = idx + 1
        num_1 = str(img_id).zfill(2)
        num_2 = str(img_id+1).zfill(2)
        imfile1 = args.img_path + num_1 + '.png'
        imfile2 = args.img_path + num_2 + '.png'

        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        # image1 = image1[None].cuda()
        # image2 = image2[None].cuda()

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre = model(image1, image2)
        
        flow_pre = padder.unpad(flow_pre[0]).cpu()[0]
        flow = flow_pre.permute(1, 2, 0).cpu().detach().numpy()
        img_flow = flow_viz.flow2rgb(flow, max_value=255.0)
        imwrite(flow_save_path + num_1+'_flow.png', img_flow)
        



    