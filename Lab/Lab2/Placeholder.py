'''
Created on 2017. 12. 5.

@author: baum-work
'''


import tensorflow as tf

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])


hypothesis = X * W + b

# reduce_mean --> 평균 계산
# square --> 제곱
cost = tf.reduce_mean(tf.square(hypothesis - Y));


# cost 최소화
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost);


sess = tf.Session()

# Variable 을 실행하기 전엔 global_variables_initializer() 를 한번 실행 해 줘야 함 
sess.run(tf.global_variables_initializer())


for step in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train], feed_dict={X:[1, 2, 3, 4, 5], Y:[2.1, 3.1, 4.1, 5.1, 6.1]})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


print("== 학습 완료 ==")

print(sess.run(hypothesis, feed_dict={X:[5]}))
print(sess.run(hypothesis, feed_dict={X:[2.5]}))
print(sess.run(hypothesis, feed_dict={X:[1.5, 3.5]}))
