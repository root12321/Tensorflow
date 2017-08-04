import tensorflow as tf
from PIL import Image
import numpy as np
img =Image.open("D:\\opencv_camera\\digits\\3.jpg")
if img.size!=(28,28):
    img= img.resize((28,28))

if img.mode!='L':
       img = img.convert('L')

def normalizepic(pic):
    im_arr = list(pic.getdata())
    im_nparr = []
    for x in im_arr:
        x=1-x/255
        im_nparr.append(x)
    im_nparr = np.array([im_nparr])
    return im_nparr
img=normalizepic(img).reshape((1,784))
img= img.astype(np.float32)
#print(normalizepic(img))
data = img
x = tf.placeholder(tf.float32, [None,784])
W = tf.Variable(tf.zeros([784,10]),dtype=tf.float32,name='weights')
b = tf.Variable(tf.zeros([10]),dtype=tf.float32,name='biases')
y = tf.nn.softmax(tf.matmul(x, W) + b)
sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "my_net/save_net.ckpt")
result = sess.run(y, feed_dict={x: data})
print(result)