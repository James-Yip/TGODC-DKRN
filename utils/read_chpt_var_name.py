model_dir = './save/neural/keyword_1'
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
reader = pywrap_tensorflow.NewCheckpointReader(model_dir)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
     print("tensor_name: ", key)