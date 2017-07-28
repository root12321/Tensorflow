import tensorflow as tf
import numpy as np
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3
#创建tensflow结构
Weights=tf.Variable(tf.random_uniform([1],-1,1))#创建一个一维的在-1~1之间的随机数
biases=tf.Variable(tf.zeros([1]))#创建一个一维的0

y=Weights*x_data+biases

loss=tf.reduce_mean(tf.square(y-y_data))#计算预测值和真实值之间的误差
optimizer=tf.train.GradientDescentOptimizer(0.5)#生成一个学习器，学习效率为0.5
train=optimizer.minimize(loss)#用来训练数据，减小误差
init=tf.initialize_all_variables()#初始化所有变量

#初始化结构
sess=tf.Session()
sess.run(init)#激活init
for step in range(201):#训练201次
    sess.run(train)#开始训练
    if step%20==0:
        print(step,sess.run(Weights),sess.run(biases))#打印出每隔20步的训练结果 输出Weights和biases