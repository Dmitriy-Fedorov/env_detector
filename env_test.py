import os
import cv2
import json
import time
from pprint import pprint
from env_detector.env import env
import fnmatch
font = cv2.FONT_HERSHEY_SIMPLEX


def write_text(frame, text):
    font_scale = 1
    text_offset_x, text_offset_y = (20,30)
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)
    cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
    return frame

if __name__ == "__main__":
    outDictionary = {}
    videoPath = "/mnt/623E39233E38F221/Users/dima_/Google Диск (dmitriy.fedorov@nu.edu.kz)/share"  # fixed for debugging purposes
    save_path = 'test'
    outDictionary["output"] = []
    
    env_ = env()  # class object that handles environment detection
    frame_counter = 1
    gstart = time.time()  # to measure Total runtime

    filelist = []
    for root, dirs, files in os.walk(videoPath):
        filelist = filelist + [os.path.join(root,x) for x in files if x.endswith(('.png'))] 
    filelist = filelist[::4]
    totalFrameNumber = len(filelist)
    for image in filelist:
        # print(image)
        start = time.time()  # for performance measurement
        frame = cv2.imread(image)
        frame_dict = {'frame_num': frame_counter}
        # frame = frame[...,::-1]  #convert BRG to RGB, due to inner working of OpenCV
        frame_dict['ENV_detector'] = env_.frame2dict(frame[...,::-1] )
        text = f"{frame_dict['ENV_detector']['ENV']} {frame_dict['ENV_detector']['ENV_confidence']}"
        res = cv2.resize(frame,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_AREA)
        res = write_text(res, text)
        outDictionary["output"].append(frame_dict)
        cv2.imwrite(f'{save_path}/{frame_counter}.png', res)
        # utility functions
        end = time.time()
        dt = end-start
        print(f'Frame {frame_counter}/{totalFrameNumber} Time: {(end-gstart)/60:.1f}m {dt:.3f}s {1/dt:.1f} fps')
        frame_counter += 1
        # if frame_counter > 6: # for debugging
        #     break
   
    # pprint(outDictionary)
    # with open('output.json', 'w') as fp:
    #     json.dump(outDictionary, fp, indent="  ")  
    # cv2.destroyAllWindows()