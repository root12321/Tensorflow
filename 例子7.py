import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer


digits=load_digits()
X=digits.data
y=digits.target
y=LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
def add_layer(inputs,in_size,out_size,layer_name,activation_function=None):
    Weight=tf.Variable(tf.random_normal([in_size,out_size]))#定义一个随机矩阵变量，行数为in_size,列数为out_size
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)#定义一个一行，out_put列的元素大小为0.1的矩阵
    Wx_plus_b=tf.matmul(inputs,Weight)+biases#做矩阵的乘法
    Wx_plus_b=tf.nn.dropout( Wx_plus_b,keep_prob)#过拟合下，用dropout决定数据保留的比例，keep_prob表示Wx_plus_b每个数据留下的概率。
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name+'/outputs',outputs)#必须有这段代码，使用tf.summary.histogram直接记录变量var的直方图，输出带直方图的汇总的protobuf
    return outputs
keep_prob=tf.placeholder(tf.float32)
xs=tf.placeholder(tf.float32,[None,64])
ys=tf.placeholder(tf.float32,[None,10])
#添加神经层
l1=add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)#定义隐藏层，输入为8*8=64个数据，输出为100个神经元
prediction=add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)#输出层


cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))#相当于是loss
tf.summary.scalar('loss',cross_entropy)#对标量数据汇总和记录使用，统计损失率
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)#训练函数，降低cross_entropy（loss）
sess=tf.Session()
summary_op=tf.summary.merge_all()#这个函数不是很明白
train_writer=tf.summary.FileWriter('logs/train',sess.graph)
test_writer=tf.summary.FileWriter('logs/test',sess.graph)
saver=tf.train.Saver()
sess.run(tf.global_variables_initializer())
for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})
    if i%50==0:
        train_result=sess.run(summary_op,feed_dict={xs:X_train,ys:y_train,keep_prob:1})
        test_result = sess.run(summary_op, feed_dict={xs:X_test, ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
save_path = saver.save(sess, "my_net/save_net.ckpt")
##train_data：被划分的样本特征集
#train_target：被划分的样本标签
#test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
#random_state：是随机数的种子。
#随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
#随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规则：
#种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数