import os
import numpy as np
import argparse
import sys
sys.path.append("./acllite")

import time

import cv2
from PIL import Image
import imageio

import matplotlib.pyplot as plt

from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel
from acllite.acllite_imageproc import AclLiteImageProc
from acllite.acllite_image import AclLiteImage

from utils.prep_utils import heatmaps_to_coordinates, COLORMAP

import pdb

def main(model_path, video_path, output_dir):
    
    # 1. initialize acl runtime 
    acl_resource = AclLiteResource()
    acl_resource.init()

    start_time = time.time()	
	# use allocated resource to load model
    model = AclLiteModel(model_path)

    # read video
    video = cv2.VideoCapture(video_path)
    frame_list = []
    ret, img_original = video.read()
    # print(img_original.shape)
    h, w, _ = img_original.shape
    # pdb.set_trace()

    counter = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames: ", total_frames)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_out = cv2.VideoWriter('./output_video.mp4',  fourcc, 30, (w, h), True)

    # preprocess image, predict and draw
    while ret:
        counter += 1
        print("Frame Capture:", counter, " / ", total_frames)
        image = img_original.copy()
        processed_input = preprocess(image)
        processed_input = processed_input[None, :, :, :]

        pred_heatmaps = model.execute([processed_input.copy()])
        pred_keypoints = postprocess(pred_heatmaps)

        draw_skeleton(pred_keypoints[0], img_original, w)

        # frame_list.append(img_original)
        # video_out.write(img_original)
        frame_list.append(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ret, img_original = video.read()
    
    # video_out.release()
    print("image in list: ", len(frame_list))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(output_dir, fourcc, 20, (w, h), True)

    end_time = time.time()
    print("total time: ", end_time - start_time)
    print("fps:", total_frames / (end_time - start_time))

    imageio.mimsave(output_dir, frame_list, 'GIF', duration=0.05)
    # duration: the time gap (s) of two images in gif


def preprocess(image):
	# do preprocessing here...
    image = cv2.resize(image, (128, 128))
    
    image = np.asarray(image) / 255
    image = np.transpose(image, [2, 0, 1])
    means = [[[0.3950]], [[0.4323]], [[0.2954]]]
    stds = [[[0.1966]], [[0.1734]], [[0.1836]]]
    image = (image - means) / stds
    image = np.asarray(image, dtype=np.float32)
	
    return image


def postprocess(pred_heatmaps):
    pred_heatmaps = np.asarray(pred_heatmaps)[0]
    pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
    return pred_keypoints


def visualize(orig_image, pred_keypoints, output_dir):
    plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
    pred_keypoints = pred_keypoints * 4344
    for finger, params in COLORMAP.items():
        plt.plot(
            pred_keypoints[params["ids"], 0],
            pred_keypoints[params["ids"], 1],
            params["color"]
        )
    plt.title("Predict")
    plt.savefig(output_dir)

def draw_skeleton(pred_keypoints, image_original, img_size):
    pred_keypoints = np.round(pred_keypoints * img_size)
    pred_keypoints = np.asarray(pred_keypoints, dtype=np.int32)

    for finger, params in COLORMAP.items():
        for i in range(len(pred_keypoints[params["ids"]])-1):
            cv2.line(image_original, 
            pred_keypoints[params["ids"]][i], 
            pred_keypoints[params["ids"]][i+1], 
            params["color"],
            3)

		
if __name__ == '__main__':
	model_path = 'model/simple2D_retrain.om'
	video_path = 'data/video/720test_video.mp4'
	output_path = './output_video.gif'
	main(model_path, video_path, output_path)
