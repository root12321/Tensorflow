import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
##这是一个添加神经层的函数
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))#定义一个随机矩阵变量，行数为in_size,列数为out_size
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#定义一个一行，out_put列的元素大小为0.1的矩阵
    Wx_plus_b=tf.matmul(inputs,Weight)+biases#做矩阵的乘法
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

x_data=np.linspace(-1,1,300)[:,np.newaxis]#在-1~1之间产生300个平均数，然后增加维度，变成300行一列的矩阵
noise=np.random.normal(0,0.05,x_data.shape)#在0~0.05之间产生一个300行一列的随机矩阵，作为噪点
y_data=np.square(x_data)-0.5+noise#在原来的计算基础上加上噪点，使他更和实际相匹配。
xs=tf.placeholder(tf.float32,[None,1])#表示输出为1，输入随机,一定要定义placeholder的类型，不然会报错。
ys=tf.placeholder(tf.float32,[None,1])
##定义隐藏层
l1=add_layer(xs,1,10,activation_function=tf.nn.relu)#隐藏层的输入为xs的size=1,输出定义为10个神经元(xs在最低下已经赋给x_data,x_data的size=1)
##定义输出层
prediction=add_layer(l1,10,1,activation_function=None)#输出层的输入为l1的size=10，输出定义为1个神经元

loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=1))#对输出层的数据和实际数据的差的平方求和再求平均

train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)#选择一个训练方式，学习效率为0.1，目的在于每次学习降低loss

init=tf.initialize_all_variables()#初始化变量
sess=tf.Session()#定义session
sess.run(init)


fig=plt.figure()#建造一个图片框
ax=fig.add_subplot(1,1,1)#编号为（1,1,1）,显示的是绘图框的大小
ax.scatter(x_data,y_data)#将图像以点的形式画出
plt.interactive(True)#目的是让图片在ax画框里连续绘制。
plt.show()#显示图像


for i in range(1001):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})#循环1000次，给train_step学习1000次降低loss
    if i%50==0:
        ##print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))#打印出学习了1000次后每隔50次的loss
        try:
            ax.lines.remove(lines[0])#因为每隔50步会画一条线，线条会有很多，所以我只需要最后一次学习的结果的那条线，每次都抹除上一条线，只保留最后一条。
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data, ys: y_data})  # 将输出层的数据保存在prediction_value中
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # 将输出层的数据画在图上，颜色为红色，线宽为5

        plt.pause(0.1)#暂停0.1秒在画下一条线
plt.pause(1000)#目的是让画面停留在最后一张图上，如果直接关闭图片，会报错。（因为我不知道那个命令可以让图片直接停在最后一个画面上，就用了这个很蠢得办法）