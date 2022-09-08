import tensorflow as tf
num_gpus=len(tf.config.list_physical_devices('GPU'))
assert num_gpus>0, "NO GPU found"
print("gpu is available")
