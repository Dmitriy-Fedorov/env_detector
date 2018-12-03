from threading import Thread
import sys
import cv2
 
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
 
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=128, start_frame=0, step_frame=0):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stream.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        self.stream.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.stopped = False
        self.step_frame = step_frame
        self.frame_num = start_frame

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
    
    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    
    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
 
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                self.stream.set(cv2.CAP_PROP_POS_FRAMES, self.frame_num)
                
                (grabbed, frame) = self.stream.read()
                
 
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return
 
                # add the frame to the queue
                self.Q.put((self.frame_num, frame))
                self.frame_num += self.step_frame

    def read(self):
        # return next frame in the queue
        print(self.Q.qsize())
        return self.Q.get()
    
    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True