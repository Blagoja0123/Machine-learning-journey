import tensorflow as tf

print(tf.version)

# 0 tf variables, rank 0, scalar
string = tf.Variable("new String", tf.string)
number = tf.Variable(32, tf.int32)
decimal = tf.Variable(3.14, tf.float32)
# print(string, number, decimal)

# rank/degree
rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["Test", "Best"], ["Test", "Best"]], tf.string)

# print(tf.rank(rank1_tensor))
# print(tf.rank(rank2_tensor))
# print(rank2_tensor.shape)

# change shape
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor1, [3, -1])
# print(tensor1)
# print(tensor2)
# print(tensor3)

# slicing

matrix = [[1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5],
          [1, 2, 3, 4, 5]]

tensor = tf.Variable(matrix, tf.int32)

# print(tensor[0,2], tensor[3], tensor[3][4])