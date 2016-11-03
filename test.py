import tensorflow as tf
import numpy as np
from PIL import Image
from TPS_STN import TPS_STN

img = np.array(Image.open("original.png"))
shape = list(img.shape)
out_size = shape

nx = 2
ny = 2
iteration = 10000

v = np.array([
  [0.1, 0.1],
  [0.2, 0.2],
  [0.2, 0.2],
  [0.4, 0.4]])

p = tf.constant(v.reshape([1, nx*ny, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape([1,shape[0],shape[1],1]), dtype=tf.float32)
t_img = TPS_STN(t_img, nx, ny, p, out_size)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  img1 = sess.run(t_img)
  Image.fromarray(np.uint8(img1.reshape(shape))).save("transformed.png") 
