import tensorflow as tf
import numpy as np
from PIL import Image
from TPS_STN import TPS_STN

img = np.array(Image.open("original.png"))
shape = list(img.shape)
out_size = shape

nx = 4
ny = 4
iteration = 10000

p = tf.constant(np.ones([1, nx*ny, 2])*0.5, dtype=tf.float32)
t_img = tf.constant(img.reshape([1,shape[0],shape[1],1]), dtype=tf.float32)
t_img = TPS_STN(t_img, 4, 4, p, out_size)

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  img1 = sess.run(t_img)
  Image.fromarray(np.uint8(img1.reshape(shape))).save("transformed.png") 
