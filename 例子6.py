import tensorflow as tf
import numpy as np
def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weight'):
            Weight=tf.Variable(tf.random_normal([in_size,out_size]),name='W')#定义一个随机矩阵变量，行数为in_size,列数为out_size
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='biases')#定义一个一行，out_put列的元素大小为0.1的矩阵
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b=tf.matmul(inputs,Weight)+biases#做矩阵的乘法
        if activation_function is None:
            outputs=Wx_plus_b
        else:
            outputs=activation_function(Wx_plus_b)
        return outputs
with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')#表示输出为1，输入随机,一定要定义placeholder的类型，不然会报错。
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')
##定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)#隐藏层的输入为xs的size=1,输出定义为10个神经元(xs在最低下已经赋给x_data,x_data的size=1)
##定义输出层
prediction=add_layer(l1,10,1,activation_function=None)#输出层的输入为l1的size=10，输出定义为1个神经元
with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=1))#对输出层的数据和实际数据的差的平方求和再求平均
with tf.name_scope('train_step'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#选择一个训练方式，学习效率为0.1，目的在于每次学习降低loss

init=tf.initialize_all_variables()#初始化变量
sess=tf.Session()#定义session
writer=tf.summary.FileWriter("logs/",sess.graph)

sess.run(init)
