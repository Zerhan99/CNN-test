import tensorflow as tf
# Tensorflow提供了一个类来处理MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
import time
 
# 载入数据集
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# 设置批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size
 
def make_weigth(shape):
    init = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(init)
 
def make_bias(shape):
    init = tf.constant(0.1,shape=shape)
    return tf.Variable(init)
"""
strides=[b,h,w,c]
b表示在样本上的步长默认为1，也就是每一个样本都会进行运算。
h表示在高度上的默认移动步长为1，这个可以自己设定，根据网络的结构合理调节。
w表示在宽度上的默认移动步长为1，这个同上可以自己设定。
c表示在通道上的默认移动步长为1，这个表示每一个通道都会进行运算
"""
# 卷积层
def make_cov2(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')
#池化层
def make_pool(value):
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
 
#输入层
#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
 
# 改变x的格式转为4维的向量[batch,in_hight,in_width,in_channels]
x_image = tf.reshape(x, [-1, 28, 28, 1])
 
#生成一个卷积
w_cov1 = make_weigth([5,5,1,32]) #生成5*5的窗口，32个卷积核从一个平面获取特征进行计算
b_cov1 = make_bias([32])#每一个卷积核有一个b
#进行卷积计算，并且使用rule激活函数
h_cov1 = tf.nn.relu(make_cov2(x_image,w_cov1)+b_cov1)
#进行池化计算
h_pool = make_pool(h_cov1)
 
#生成第二个卷积
w_cov2 = make_weigth([5,5,32,64])
b_cov2 = make_bias([64])
h_cov2 = tf.nn.relu(make_cov2(h_pool,w_cov2)+b_cov2)
h_pool2 = make_pool(h_cov2)
 
#实现全连接层
w_fc1 = make_weigth([7*7*64,128])
b_fc1 = make_bias([1,128])
 
# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
# 求第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
 
# keep_prob: float类型，每个元素被保留下来的概率，设置神经元被选中的概率,在初始化时keep_prob是一个占位符
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
 
# 实现第二个全连接层
w_fc2 = make_weigth([128,10])
b_fc2 = make_bias([1,10])
#求去输出 不需要进行激活函数
# 输出层
# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
 
# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 求准确率(tf.cast将布尔值转换为float型)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# 开始运行
with tf.Session() as sess:
    start_time = time.clock()
    sess.run(tf.global_variables_initializer()) #初始化所有的变量
    #进行迭代5次迭代
    for epoch in range(5):
        for batch in range(n_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y,keep_prob:0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print('Iter' + str(epoch) + ',Testing Accuracy=' + str(acc))
    end_time = time.clock()
    print('Running time:%s Second' % (end_time - start_time))  # 输出运行时间