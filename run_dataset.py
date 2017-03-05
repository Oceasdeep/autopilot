import tensorflow as tf
import scipy.misc
import cv2
import time
import logger
import os
from tensorflow.python.platform import gfile
from tensorflow.core.framework import graph_pb2

MODELDIR = './save'
GRAPHFILE = 'frozen_graph.pb'
LOGDIR = './logs'
RESULTSDIR = './results'
RESULTFILE = 'run.py.csv'
DATASETDIR = './driving_dataset/scaled'


#saver = tf.train.Saver()
#saver.restore(sess, os.path.join(MODELDIR,'model.ckpt'))

# Create session and restore loaded graph
sess = tf.InteractiveSession()
sess.graph.as_default()


# Load graph def from a protobuf file
graph_def = graph_pb2.GraphDef()
with open(os.path.join(MODELDIR, GRAPHFILE) , "r") as f:
    graph_def.ParseFromString(f.read())


tf.import_graph_def(graph_def)

graph = tf.get_default_graph()

# init = tf.initialize_all_variables()
# sess.run(init)

# Recover placeholders and tensors
# TensorFlow seems to namespace the imported tensors with "import/"
x = graph.get_tensor_by_name("import/x:0")
keep_prob = sess.graph.get_tensor_by_name("import/keep_prob:0")
y = graph.get_tensor_by_name("import/y:0")

# Initialize variables
smoothed_angle = 0.0
degrees = 0.0
i = -1
t = time.time()
t_prev = t
t0 = t


# Create ResultLogger and write initial log entry
if not os.path.exists(RESULTSDIR):
    os.makedirs(RESULTSDIR)
log = logger.ResultLogger(os.path.join(RESULTSDIR, RESULTFILE))
log.write(i, t-t0, 0.0, degrees, smoothed_angle)

while(True): #cv2.waitKey(10) != ord('q')):

    i += 1

    # Read 66 x 200 RGB image
    try:
        full_image = scipy.misc.imread(DATASETDIR + "/" + str(i) + ".jpg", mode="RGB")
    except:
        break

    # Resize to 66 x 200 and scale to interval [0,1]
    normalized_image = full_image / 255.0

    # Perform inference
    degrees = sess.run(y, feed_dict={x: [normalized_image], keep_prob: 1.0})[0][0] * 180.0 / scipy.pi

    # make smooth angle transitions by turning the steering wheel based on the difference of the current angle
    # and the predicted angle
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)

    # Measure current time
    t = time.time()

    # Write to driving log
    log.write(i, t-t0, t-t_prev, degrees, smoothed_angle)
    t_prev = t
