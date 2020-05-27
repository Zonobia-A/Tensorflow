# 实例，以逻辑回归拟合二维数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 平滑loss曲线
def moving_average(a, w=100):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w): idx]) / w for idx, val in enumerate(a)]

# 准备数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()
# 搭建模型
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.get_variable("weight", shape=[1], initializer=tf.random_normal_initializer)
# w = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.zeros([1]), name="bias")
b = tf.get_variable("bias", shape=[1], initializer=tf.random_normal_initializer)
# 正向
z = tf.multiply(X, w) + b
# 反向
cost = tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义超参数
train_epochs = 200
display_step = 2
# 启动Session
with tf.Session() as sess:
    sess.run(init)
    plotdata = {"batchsize": [], "loss": []}
    # 向模型输入数据
    for epoch in range(train_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        # 显示训练的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1,
                  "cost=", loss,
                  "W=", sess.run(w),
                  "b=", sess.run(b))
            if not (loss == "NA"):
                plotdata["batchsize"].append(epoch)
                plotdata["loss"].append(loss)
    print("Finished!")
    print("cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}),
          "W=", sess.run(w),
          "b=", sess.run(b))
    # 图形显示
    plt.plot(train_X, train_Y, 'ro', label='original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label='Fittedline')
    plt.legend()
    plt.show()
    plotdata['avgloss'] = moving_average(plotdata['loss'])
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata['batchsize'], plotdata['avgloss'], 'b--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
    # Test data
    print("x = 0.5, y = ", sess.run(z, feed_dict={X: 0.5}))
