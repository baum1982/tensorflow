import tensorflow as tf
from tensorflow.contrib.labeled_tensor import placeholder

hello = tf.constant("Hello, Tensorflow!")
sess = tf.Session()
print(sess.run(hello))





node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)


sess = tf.Session()

sess.run([node1,node2])

print("sess.run([node1,node2]): ", sess.run([node1,node2]))
print("sess.run(node3): ", sess.run(node3))






a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b # + provides a shortcut for df.add(a,b)


print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))


