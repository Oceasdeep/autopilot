"""NVIDIA end-to-end deep learning inference for self-driving cars.

This script loads a pretrained graph from a graph def protobuf file and
performs inference based on that graph using jpeg images
as input and produces an output of steering wheel angle as propotions of a
full turn.
"""
import tensorflow as tf
import scipy.misc
import cv2
import time
import logger
import os
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

# Folder locations
MODELDIR = './save'
GRAPHFILE = 'frozen_graph.pb'
LOGDIR = './logs'
RESULTSDIR = './results'
RESULTFILE = 'run.py.csv'
DATASETDIR = './driving_dataset/scaled'

# Create session and restore loaded graph
sess = tf.InteractiveSession()
sess.graph.as_default()

# Load graph def from a protobuf file
graph_def = graph_pb2.GraphDef()
with open(os.path.join(MODELDIR, GRAPHFILE) , "r") as f:
    graph_def.ParseFromString(f.read())

tf.import_graph_def(graph_def)
graph = tf.get_default_graph()

# Recover placeholders and tensors
# TensorFlow seems to namespace the imported tensors with "import/"
x = graph.get_tensor_by_name("import/x:0")
keep_prob = sess.graph.get_tensor_by_name("import/keep_prob:0")
y = graph.get_tensor_by_name("import/y:0")

# Initialize variables used during the inference loop
smoothed_angle = 0.0
output = 0.0
i = -1
t = time.time()
t_prev = t
t0 = t

# Create ResultLogger and write initial log entry
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)
log = logger.ResultLogger(os.path.join(RESULTSDIR, RESULTFILE))
log.write(i, t-t0, 0.0, output)

# Inference loop. Run through all the images in the dataset and perform
# inference.
while(True):

    # Increment image index
    i += 1

    # Read the next 66 x 200 RGB image
    try:
        full_image = scipy.misc.imread(DATASETDIR + "/" + str(i) + ".jpg", mode="RGB")
    except:
        break

    # Normalize image
    normalized_image = full_image / 255.0

    # Perform inference
    output = sess.run(y, feed_dict={x: [normalized_image], keep_prob: 1.0})[0][0]

    # Measure current time
    t = time.time()

    # Write to driving log
    log.write(i, t-t0, t-t_prev, output)
    t_prev = t
