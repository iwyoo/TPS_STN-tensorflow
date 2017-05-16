# TPS_STN-tensorflow
TensorFlow implementation of Thin Plate Spline Spatial Transformer Network

```python
# test.py
v = np.array([
  [0.2, 0.2],
  [0.4, 0.4],
  [0.6, 0.6],
  [0.8, 0.8]])

p = tf.constant(v.reshape([1, nx*ny, 2]), dtype=tf.float32)
t_img = tf.constant(img.reshape(shape), dtype=tf.float32)
t_img = TPS_STN(t_img, nx, ny, p, out_size)
```

![alt tag](original.png)
![alt tag](transformed.png)

## References
- [Robust Scene Text Recognition with Automatic Rectification](https://arxiv.org/abs/1603.03915)
- [Spatial Transformer Network](https://arxiv.org/abs/1506.02025)
- [TensorFlow STN implementation](https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py)
