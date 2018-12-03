import os
import cv2
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pprint import pprint
from helpers.defaults import argHandler


FLAGS = argHandler()
FLAGS.setEnvDefaults()
FLAGS.parseArgs(sys.argv)

if FLAGS.yolo:
    from yolo.yolo import yolo
if FLAGS.env:
    from env_detector.env import env
    env_ = env()  # class object that handles environment detection


def init_json_dict(cap):
    outDictionary = {}
    outDictionary['videoPath'] = FLAGS.input
    outDictionary['totalFrameNumber'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outDictionary['fps'] = int(cap.get(cv2.CAP_PROP_FPS))  # float objects are not json serializable
    outDictionary["output"] = []
    return outDictionary

def skip_frame(cap):
    if FLAGS.skip_frame != 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(1) + FLAGS.skip_frame - 1)

if __name__ == "__main__":
    
    # FLAGS.input = "/mnt/623E39233E38F221/Users/dima_/Google Диск (dmitriy.fedorov@nu.edu.kz)/share/10.02.18 Muir Beach Overlook/107ORBIV/107ORBIV.CFGCILINDRIC.mp4"
    cap = cv2.VideoCapture(FLAGS.input)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FLAGS.start_frame)
    totalFrameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outDictionary = init_json_dict(cap)
        
    frame_counter = 1
    gstart = time.time()  # to measure Total runtime

    while(cap.isOpened()):
        start = time.time()  # for performance measurement
        frame_num = cap.get(1)
        frame_dict = {'frame_num': frame_num}
        ret, frame = cap.read()
        skip_frame(cap)

        if ret == True:
            frame_dict['ENV_detector'] = env_.frame2dict(frame)
        else:
            break
        
        outDictionary["output"].append(frame_dict)
        # utility functions
        end = time.time()
        dt = end-start
        if FLAGS.verbose:
            print(f'Frame {int(frame_num)}/{totalFrameNumber} Time: {(end-gstart)/60:.1f}m {dt:.3f}s {1/dt:.1f} fps')
        
        if frame_counter == FLAGS.end_frame: # for debugging
            break
        frame_counter += 1
    print(f"Total time: {(time.time()-gstart)}")
    env_.sess.close()
    cap.release()
    # pprint(outDictionary)
    with open(FLAGS.output, 'w') as fp:
        json.dump(outDictionary, fp, indent="  ")

    if FLAGS.debug:
        from helpers.DebugVideo import DebugVideo
        DebugVideo(FLAGS.output, FLAGS.debug_out).write()
    