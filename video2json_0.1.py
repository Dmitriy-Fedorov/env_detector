import os
import cv2
import json
import time
from pprint import pprint
from env_detector.env import env
from yolo.yolo import yolo




if __name__ == "__main__":
    outDictionary = {}
	# videoPath = input("Enter the path to video: ")
    videoPath = "/home/dfed/Desktop/vlog.mp4"  # fixed for debugging purposes
    cap = cv2.VideoCapture(videoPath)
    totalFrameNumber = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    outDictionary['totalFrameNumber'] = int(totalFrameNumber)
    outDictionary['fps'] = int(cap.get(cv2.CAP_PROP_FPS))  # float objects are not json serializable
    outDictionary["output"] = []

    env_ = env()  # class object that handles environment detection
    yolo_ = yolo()  # class object that handles yolo
    frame_counter = 1
    gstart = time.time()  # to measure Total runtime

    while(cap.isOpened()):
        start = time.time()  # for performance measurement
        ret, frame = cap.read()
        frame_dict = {'frame_num': frame_counter}

        if ret == True:
            frame = frame[...,::-1]  #convert BRG to RGB, due to inner working of OpenCV
            frame_dict['ENV_detector'] = env_.frame2dict(frame)
            frame_dict["YOLO"] = yolo_.frame2dict(frame)
        else:
            break

        if frame_counter > 5: # for debugging
            break
        
        outDictionary["output"].append(frame_dict)
        # utility functions
        end = time.time()
        dt = end-start
        print(f'Frame {frame_counter}/{totalFrameNumber} Time: {(end-gstart)/60:.1f}m {dt:.3f}s {1/dt:.1f} fps')
        frame_counter += 1
    
    cap.release()
    # pprint(outDictionary)
    with open('output.json', 'w') as fp:
        json.dump(outDictionary, fp, indent="  ")    

