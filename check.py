from tensorflow.python import pywrap_tensorflow
import os
checkpoint_path = os.path.join("./save/neural_mask_keyword_predictor_gcndrk8", "keyword_predictor_1")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path) #tf.train.NewCheckpointReader
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
