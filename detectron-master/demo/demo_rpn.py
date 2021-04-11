# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import ipdb
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
import numpy as np
from predictor_rpn import VisualizationDemo
import torch.nn as nn
# constants
WINDOW_NAME = "COCO detections"
import torch
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify model config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
    
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence

def get_nonzero_std(s):
    # `s` has shape (T, V, C)
    # Select valid frames where sum of all nodes is nonzero
    s = s[s.sum((1,2)) != 0]
    if len(s) != 0:
        # Compute sum of standard deviation for all 3 channels as `energy`
        s = s[..., 0].std() + s[..., 1].std() + s[..., 2].std()
    else:
        s = 0
    return s

def read_xyz(file, max_body=4, num_joint=25):  # 
    seq_info = read_skeleton_filter(file)
    # Create data tensor of shape: (# persons (M), # frames (T), # nodes (V), # channels (C))
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['colorX'], v['colorY'], v['z']]

    # select 2 max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]
    # Data new shape: (C, T, V, M)
    data = data.transpose(3, 1, 2, 0)
    return data


def get_res_box(file,i,aaa):

    man10 = []
    man06 = []
    man10.append(aaa[:,i,10,0])
    man10.append(aaa[:,i,3,0])
    man10.append(aaa[:,i,19,0])
    man06.append(aaa[:,i,6,0])
    man06.append(aaa[:,i,15,0]) 
    res_box10 = [[man10[0][0]-70,man10[0][1]-70,man10[0][0]+100,man10[0][1]+100],[man10[1][0]-80,man10[1][1]-80,man10[1][0]+80,man10[1][1]+80],[man10[2][0]-120,man10[2][1]-100,man10[2][0]+120,man10[2][1]+100]] 
    res_box06 = [[man06[0][0]-70,man06[0][1]-70,man06[0][0]+70,man06[0][1]+100],[man06[1][0]-120,man06[1][1]-100,man06[1][0]+120,man06[1][1]+100]]

    return res_box10, res_box06

def get_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            break

def get_feature(video_path, result_path, cfg):
    print(video_path)
    name = video_path[-28:-8]
    skeleton_name = name[0:20]
    category = skeleton_name[-4:]
    isExists = os.path.exists("/dataset/zll/NTU/box/"  + category + '/')
    if not isExists:
        os.makedirs("/dataset/zll/NTU/box/" + category + '/')
    
    
    skeleton_file = "/data/shuju/dataset/nturgbd_skeletons_120/" + skeleton_name + '.skeleton'
    
    #ipdb.set_trace()
    
    output = torch.zeros(1,256).cuda()
    
    i = 0
    demo = VisualizationDemo(cfg)
    aaa = read_xyz(skeleton_file)
    
    
    for image in get_frames(video_path):
        
        res_box10, res_box06 = get_res_box(skeleton_file, i, aaa)
        
        predictions, boxes_center = demo.run_on_image(image, res_box06, res_box10)   # (256,7,7)
        
        #category = skeleton_name[-4:]
        rpn_path = "/dataset/zll/NTU/box/" + category + '/' + skeleton_name + '.txt'
        f_rpn = open(rpn_path, 'a')
        for i in range(3):
            f_rpn.write('{},{},{},{}\n'.format(boxes_center[i][0], boxes_center[i][1], boxes_center[i][2], boxes_center[i][3]))
        f_rpn.close()
        
        m = nn.AdaptiveAvgPool2d((1,1))
        #ipdb.set_trace()
        outputm_0 = m(predictions[0])
        outputm_1 = m(predictions[1])
        outputm_2 = m(predictions[2])
        f_0 = torch.squeeze(outputm_0)
        f_0 = torch.unsqueeze((f_0),0)
        f_1 = torch.squeeze(outputm_1)
        f_1 = torch.unsqueeze((f_1),0)
        f_2 = torch.squeeze(outputm_2)
        f_2 = torch.unsqueeze((f_2),0)
        #output.append(zf[2])
        output = torch.cat((output,f_0),0)
        output = torch.cat((output,f_1),0)
        output = torch.cat((output,f_2),0)   #(4,256)     
        if i == 255:
            break
        i += 1
    
    #ipdb.set_trace()
    b = output[1:] #(3*256,256)
    b = b.cpu().detach().numpy()
    s = np.transpose(b,[1,0]) #(256,3*256)
    personzero = np.zeros((256,256*3))
    size = s.shape
    if size[1] <= 256*3:
        personzero[:,0:size[1]] = s
    else:
        personzero = s[:,0:256*3]
        
    np.save(result_path + '/' + name, personzero)
    

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    
    cfg = setup_cfg(args)

    path_root = "/dataset/NTU/NTU_RGB_120/"
   
    for classes in os.listdir(path_root):
        #if classes == '001' or classes == '002' or classes == '003':	
		classes_dir = os.path.join(path_root, classes)		
		result_path = "/dataset/zll/NTU/feature/" + 'A' + classes
		isExistss = os.path.exists(result_path)
		if not isExistss:
			os.makedirs(result_path)			
		for dirs in os.listdir(classes_dir):
			video_path = os.path.join(classes_dir,dirs)
			video_name = dirs[0:20]
				#result_path0 = result_path + '/' + video_name + '.npy'
			get_feature(video_path, result_path, cfg)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
 