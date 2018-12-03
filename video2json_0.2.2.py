import os
import cv2
import sys
import json
import time
import numpy as np
import tensorflow as tf
from pprint import pprint
from darkflow.defaults import argHandler
from helpers.FileVideoStream import FileVideoStream


FLAGS = argHandler()
FLAGS.setEnvDefaults()
FLAGS.parseArgs(sys.argv)

if FLAGS.yolo:
    from yolo.yolo import yolo
if FLAGS.env:
    from env_detector.env import env

if __name__ == "__main__":
    outDictionary = {}
    FLAGS.input = "/mnt/623E39233E38F221/Users/dima_/Google Диск (dmitriy.fedorov@nu.edu.kz)/share/10.02.18 Muir Beach Overlook/107ORBIV/107ORBIV.CFGCILINDRIC.mp4"
    fvs = FileVideoStream(FLAGS.input, queueSize=20,step_frame=5).start()
    time.sleep(5)
    cap = fvs.stream
    totalFrameNumber = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    outDictionary['totalFrameNumber'] = totalFrameNumber
    outDictionary['fps'] = int(cap.get(cv2.CAP_PROP_FPS))  # float objects are not json serializable
    outDictionary["output"] = []

    if FLAGS.env:
        env_ = env()  # class object that handles environment detection
    frame_counter = 1
    gstart = time.time()  # to measure Total runtime

    while fvs.more():
        start = time.time()  # for performance measurement
        frame_num, frame = fvs.read()
        frame_dict = {'frame_num': frame_num}
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + FLAGS.step_frame)
        frame_dict['ENV_detector'] = env_.frame2dict(frame)
        
        outDictionary["output"].append(frame_dict)
        # utility functions
        end = time.time()
        dt = end-start
        print(f'Frame {int(frame_num)}/{totalFrameNumber} Time: {(end-gstart)/60:.1f}m {dt:.3f}s {1/dt:.1f} fps')
        
        if frame_counter == FLAGS.end_frame: # for debugging
            break
        frame_counter += 1
    
    print(f"Total time: {(time.time()-gstart)}")
    env_.sess.close()
    cap.release()
    fvs.stop()
    # pprint(outDictionary)
    with open(FLAGS.output, 'w') as fp:
        json.dump(outDictionary, fp, indent="  ")   