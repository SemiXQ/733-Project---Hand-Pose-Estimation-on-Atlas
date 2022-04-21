import os
import numpy as np
import argparse
import sys
sys.path.append("./acllite")

import time

import cv2
import imageio

from acllite.acllite_resource import AclLiteResource
from acllite.acllite_model import AclLiteModel
from acllite.acllite_imageproc import AclLiteImageProc
from acllite.acllite_image import AclLiteImage

import pdb

def draw_bd_handpose(img_,hand_,x,y):
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    #
    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['1']['x']+x), int(hand_['1']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['1']['x']+x), int(hand_['1']['y']+y)),(int(hand_['2']['x']+x), int(hand_['2']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['2']['x']+x), int(hand_['2']['y']+y)),(int(hand_['3']['x']+x), int(hand_['3']['y']+y)), colors[0], thick)
    cv2.line(img_, (int(hand_['3']['x']+x), int(hand_['3']['y']+y)),(int(hand_['4']['x']+x), int(hand_['4']['y']+y)), colors[0], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['5']['x']+x), int(hand_['5']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['5']['x']+x), int(hand_['5']['y']+y)),(int(hand_['6']['x']+x), int(hand_['6']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['6']['x']+x), int(hand_['6']['y']+y)),(int(hand_['7']['x']+x), int(hand_['7']['y']+y)), colors[1], thick)
    cv2.line(img_, (int(hand_['7']['x']+x), int(hand_['7']['y']+y)),(int(hand_['8']['x']+x), int(hand_['8']['y']+y)), colors[1], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['9']['x']+x), int(hand_['9']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['9']['x']+x), int(hand_['9']['y']+y)),(int(hand_['10']['x']+x), int(hand_['10']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['10']['x']+x), int(hand_['10']['y']+y)),(int(hand_['11']['x']+x), int(hand_['11']['y']+y)), colors[2], thick)
    cv2.line(img_, (int(hand_['11']['x']+x), int(hand_['11']['y']+y)),(int(hand_['12']['x']+x), int(hand_['12']['y']+y)), colors[2], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['13']['x']+x), int(hand_['13']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['13']['x']+x), int(hand_['13']['y']+y)),(int(hand_['14']['x']+x), int(hand_['14']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['14']['x']+x), int(hand_['14']['y']+y)),(int(hand_['15']['x']+x), int(hand_['15']['y']+y)), colors[3], thick)
    cv2.line(img_, (int(hand_['15']['x']+x), int(hand_['15']['y']+y)),(int(hand_['16']['x']+x), int(hand_['16']['y']+y)), colors[3], thick)

    cv2.line(img_, (int(hand_['0']['x']+x), int(hand_['0']['y']+y)),(int(hand_['17']['x']+x), int(hand_['17']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['17']['x']+x), int(hand_['17']['y']+y)),(int(hand_['18']['x']+x), int(hand_['18']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['18']['x']+x), int(hand_['18']['y']+y)),(int(hand_['19']['x']+x), int(hand_['19']['y']+y)), colors[4], thick)
    cv2.line(img_, (int(hand_['19']['x']+x), int(hand_['19']['y']+y)),(int(hand_['20']['x']+x), int(hand_['20']['y']+y)), colors[4], thick)



def main(model_path, video_path, output_dir):
    
    # 1. initialize acl runtime 
    acl_resource = AclLiteResource()
    acl_resource.init()

    start_time = time.time()	
	# use allocated resource to load model
    model = AclLiteModel(model_path)

    img_size = 256

    # read video
    video = cv2.VideoCapture(video_path)
    frame_list = []
    ret, image_original = video.read()
    image_height, image_width, _ = image_original.shape
    # pdb.set_trace()

    counter = 0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    print("total frames: ", total_frames)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_out = cv2.VideoWriter('./output_video.mp4',  fourcc, 30, (image_width, image_height), True)

    # preprocess image, predict and draw
    while ret:
        counter += 1
        print("Frame Capture:", counter, " / ", total_frames)

        image = cv2.resize(image_original, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        image = image.transpose((2, 0, 1))
        image = image / 255.
        image = np.expand_dims(image, 0)
        image = image.astype('float32')

        # predict
        output = model.execute([image.copy()])
        output = output[0][0]
        output = np.array(output)
        # print(output.shape[0])

        pts_hand = {}
        for i in range(int(output.shape[0]/2)):
            x = (output[i*2+0]*float(image_width))
            y = (output[i*2+1]*float(image_height))

            pts_hand[str(i)] = {}
            pts_hand[str(i)] = {
                "x":x,
                "y":y,
                }

        draw_bd_handpose(image_original,pts_hand,0,0) # draw lines

        # draw points
        for i in range(int(output.shape[0]/2)):
            x = (output[i*2+0]*float(image_width))
            y = (output[i*2+1]*float(image_height))

            cv2.circle(image_original, (int(x),int(y)), 3, (255,50,60),-1)
            cv2.circle(image_original, (int(x),int(y)), 1, (255,150,180),-1)
        

        # video_out.write(image_original)
        frame_list.append(cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB))
        ret, image_original = video.read()
    
    # video_out.release()
    print("image in list: ", len(frame_list))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(output_dir, fourcc, 20, (w, h), True)

    end_time = time.time()
    print("total time: ", end_time - start_time)
    print("fps:", total_frames / (end_time - start_time))

    imageio.mimsave(output_dir, frame_list, 'GIF', duration=0.05)
    # duration: the time gap (s) of two images in gif

		
if __name__ == '__main__':
	model_path = 'model/handx.om'
	video_path = 'data/video/test_video2.mp4'
	output_path = './output_video2.gif'
	main(model_path, video_path, output_path)
