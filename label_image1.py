from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import argparse
import sys
import time

import numpy as np
import tensorflow as tf
import cv2 as c
import threading

# Author A.C.M.Sulhi (CSE,UOM)
# This Code based on concurrent programming . when the code is running two threads executes . 
# one thread is capture the photos and save it in a specific file , after insert the photo name to "queue_teken queue"
# other thread is initially load the trained tensorflow graph and if there any photo in "queue_taken" queue pop photo one by one
# and check what is the grip optimal for the object in the image. 

  
queue_taken = []      # 

class detection(threading.Thread):
  def __init__(self , lock):
    threading.Thread.__init__(self)
    self.model_file = "tf_files/retrained_graph.pb"
    self.label_file = "tf_files/retrained_labels.txt"
    self.input_height = 128
    self.input_width = 128
    self.input_mean = 128
    self.input_std = 128
    self.input_layer = "input"
    self.output_layer = "final_result"
    self.locked = lock

  def load_graph(self,model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph

  def read_tensor_from_image_file(self,file_name, input_height=299, input_width=299,
  				input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
      image_reader = tf.image.decode_png(file_reader, channels = 3,
                                         name='png_reader')
    elif file_name.endswith(".gif"):
      image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                    name='gif_reader'))
    elif file_name.endswith(".bmp"):
      image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
      image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                          name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0);
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result

  def load_labels(self,label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label

  def run(self):
    graph = self.load_graph(self.model_file)
    file_path = "captured/"
    while True:
      if self.locked.getLock():
        if len(queue_taken)!=0:
          self.locked.setLock()
          count =queue_taken.pop(0)
          print ("poped" , count)
          self.locked.releceLock()
          file_name = file_path + count
          
          t = self.read_tensor_from_image_file(file_name,
                                      input_height=self.input_height,
                                      input_width=self.input_width,
                                      input_mean=self.input_mean,
                                      input_std=self.input_std)

          input_name = "import/" + self.input_layer
          output_name = "import/" + self.output_layer
          input_operation = graph.get_operation_by_name(input_name);
          output_operation = graph.get_operation_by_name(output_name);

          with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})
            end=time.time()
          results = np.squeeze(results)

          top_k = results.argsort()[-5:][::-1]
          labels = self.load_labels(self.label_file)

          print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))
          template = "{} (score={:0.5f})"
          for i in top_k:
            print(template.format(labels[i], results[i]))

          
          
          tf.gfile.Remove(file_name)




class capturing(threading.Thread):
  def __init__(self , lock):
    threading.Thread.__init__(self)
    self.locked = lock
  def run(self):
    cap=c.VideoCapture(0)
    path = "captured/"
    count = 1
    while 1:
      x = input("Enter to Capture:")
      ret, frame = cap.read()
      print ("captured" , count)
      name1 = path+ str(count) +'.JPG'
      n = str(count)+'.JPG'
      
      if self.locked.getLock():
        self.locked.setLock()
        c.imwrite(name1,frame)
        print ("saved" , count)
        queue_taken.append(n)
        print ("appended ",count)
        self.locked.releceLock()
      count+=1
      

# we creating a lock object to provide synchronisation to shared resourses. in our code "queue_taken" queue is a shared resourse.

class Lock:
  def __init__(self):
    self.locked = True

  def setLock(self):
    self.locked=False

  def releceLock(self):
    self.locked=True

  def getLock(self):
    return self.locked

if __name__ == "__main__":
    
  
  lock = Lock()
  cap1 = capturing(lock)
  det = detection(lock)

  cap1.start()
  det.start()

  print ("finished")
  