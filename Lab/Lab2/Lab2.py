'''
Created on 2017. 12. 5.

@author: baum-work
'''



import tensorflow as tf


x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")


hypothesis = x_train * W + b

# reduce_mean --> 평균 계산
# square --> 제곱
cost = tf.reduce_mean(tf.square(hypothesis - y_train));



# cost 최소화

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost);


sess = tf.Session()


# Variable 을 실행하기 전엔 global_variables_initializer() 를 한번 실행 해 줘야 함 
sess.run(tf.global_variables_initializer())


for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))