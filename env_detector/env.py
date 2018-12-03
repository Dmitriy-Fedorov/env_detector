import numpy as np
import cv2
import tensorflow as tf
import time
import os
import numpy as np

class env:

    def __init__(self, 
         model_file=f'{os.getcwd()}/env_detector/bin/retrained_graph.pb',
         label_file=f'{os.getcwd()}/env_detector/labels/retrained_labels.txt'):
        self.graph = self.load_graph(model_file)
        self.labels = self.load_labels(label_file)
        self.input_layer = "input"
        self.output_layer = "final_result"
        self.input_name = "import/" + self.input_layer
        self.output_name = "import/" + self.output_layer
        self.input_operation = self.graph.get_operation_by_name(self.input_name);
        self.output_operation = self.graph.get_operation_by_name(self.output_name);
        self.sess = tf.Session(graph=self.graph)

    @staticmethod
    def load_graph(model_file):
        graph = tf.Graph()
        graph_def = tf.GraphDef()
        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            tf.import_graph_def(graph_def)
        return graph

    @staticmethod
    def read_tensor_from_cv2(image, input_height=224, input_width=224,
                                    input_mean=128, input_std=128):
        float_caster = tf.cast(image, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        with tf.Session() as sess:
            result = sess.run(normalized)
        return result

    @staticmethod
    def read_tensor_from_cv2_np(image, input_height=224, input_width=224,
                                    input_mean=128, input_std=128):
        float_caster = image.astype(np.float32)
        resized = cv2.resize(float_caster,(input_width, input_height), interpolation=cv2.INTER_LINEAR)
        normalized = np.divide(np.subtract(resized, [input_mean]), [input_std])
        result = np.expand_dims(normalized, 0)
        return result
    
    @staticmethod
    def load_labels(label_file):
        label = []
        proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
        for l in proto_as_ascii_lines:
            label.append(l.rstrip())
        return label

    def make_prediction(self, t):
        graph = self.graph
        labels = self.labels
        input_operation = self.input_operation
        output_operation = self.output_operation

        results = self.sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
        results = np.squeeze(results)
        top_k = results.argsort()[-5:][::-1]
        temp = (results[top_k[0]], labels[top_k[0]])
        # print(temp)
        return temp

    def frame2dict(self, frame, input_height=224, input_width=224,
                        input_mean=128, input_std=128):

        t = self.read_tensor_from_cv2_np(frame,
                                input_height=input_height,
                                input_width=input_width,
                                input_mean=input_mean,
                                input_std=input_std)
        result, label = self.make_prediction(t)
        to_return = {"ENV": label,
                     "ENV_confidence": str(result),
			         "ENV+": "OpenNature/OpenStreet/ClosedNature/ClosedHome",
			         "ENV+_confidence": 'TODO'}
        return to_return

