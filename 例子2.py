import tensorflow as tf
matrix1=tf.constant([3,3],shape=[1,2])#生成一个一行两列的矩阵
matrix2=tf.constant([[2],
                     [2]])#生成一个两行一列的矩阵
product=tf.matmul(matrix1,matrix2)#矩阵的乘法，将俩个矩阵相乘
#方式一
sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()
#方式2
with tf.Session() as sess:
    result2=sess.run(product)
    print(result2)