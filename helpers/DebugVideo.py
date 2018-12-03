import cv2
import json
from pprint import pprint


class DebugVideo:

    def __init__(self, json_path, out_path, resolution=(640,480), video_path=None):
        with open(json_path, 'r') as fp:
            self.json = json.load(fp) 
        if video_path is not None:
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(self.json['videoPath'])
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.resolution = resolution
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(out_path,fourcc, 20.0, self.resolution)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame_list = self.json['output']

    def write_text(self, frame, text):
        font = self.font
        font_scale = 1
        text_offset_x, text_offset_y = (20,30)
        (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
        cv2.rectangle(frame, box_coords[0], box_coords[1], (255,255,255), cv2.FILLED)
        cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=2)
        return frame

    def write(self):
        cap = self.cap
        out = self.out
        for frame_label in self.frame_list:
            _, frame = cap.read()
            text = f'{frame_label["ENV_detector"]["ENV"]}: {frame_label["ENV_detector"]["ENV_confidence"]:.4}'
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_LINEAR)
            frame = self.write_text(frame, text)
            out.write(frame)
        
        cap.release()
        out.release()

    