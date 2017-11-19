import tensorflow as tf
import numpy as np
import sys

# Let us now demonstrate our ability to manipulate tensors with more than 26 legs. 
# Please pass whichever dimension you want as the first argument to this script.
# But beware, dimensions bigger than 20 or so take a while :)
test_dim = int(sys.argv[1])

sess = tf.Session()
array2 = np.arange(2**test_dim).reshape([2 for i in range(test_dim)])
d = tf.constant(array2)

# now please transpose over the last two dimensions
tensor_string = ''.join(['('+str(i)+')' for i in range(test_dim)]) + ''.join(['->']+['('+str(i)+')' for i in [*range(test_dim-2),test_dim-1,test_dim-2]])

e = tf.einsum(tensor_string, d)
r = sess.run([e])[0]

# this should always print the matrix
# [[0 2]
# [1 3]]
print(r[tuple(0 for i in range(test_dim-2))])


