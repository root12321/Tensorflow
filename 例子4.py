import tensorflow as tf
input1=tf.placeholder(tf.float32)#定义一个placeholder，外部输入结果
input2=tf.placeholder(tf.float32)
output=tf.multiply(input1,input2)#输出input1和input2d的相乘结果
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[2],input2:[7]}))#feed_dict意思是以字典的形式输入input1和input2