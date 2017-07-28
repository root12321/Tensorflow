import tensorflow as tf
#tensorflow定义变量的的方式
state=tf.Variable(0,name='counter')
#print(state.name)
one=tf.constant(1)#定义一个常量1
new_value=tf.add(state,one)#add方法是加号的意思，将state和one相加。
update=tf.assign(state,new_value)#更新变量，将new_value的值赋给state完成更新
init=tf.initialize_all_variables()#初始化变量
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):#任意循环三次
        sess.run(update)
        print(sess.run(state))


        ###总结 如果有变量定义，一定要有init=tf.initialize_all_variables()初始化变量，定义了session后一定要有sess.run(init)
