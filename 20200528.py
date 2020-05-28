# tensorflow从操作层面可以抽象成两种，模型构建和模型运行
# 实例5，编写hello world程序演示session的使用
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# hello = tf.constant('Hello World')
# sess = tf.Session()
# print(sess.run(hello))
# sess.close()
# 实例6，使用with来开启session
# a = tf.constant(0.1)
# b = tf.constant(0.3)
# c = tf.add(a, b)
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run(c))


# 实例7，注入机制
# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# c = tf.add(a, b)
# d = tf.multiply(a, b)
# config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
# with tf.Session(config=config) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     print(sess.run([c, d], feed_dict={a: 1, b: 2}))


# 实例8，模型的载入与保存
# 模型训练
# train_X = np.linspace(0, 10, 100)
# train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.2
# X = tf.placeholder('float')
# Y = tf.placeholder('float')
# w = tf.get_variable('weight', shape=[1])
# b = tf.get_variable('bias', shape=[1])
# Z = tf.multiply(X, w) + b
# epochs = 50
# lr = 0.0001
# cost = tf.reduce_mean(tf.square(Y - Z))
# optimizer = tf.train.GradientDescentOptimizer(lr).minimize(cost)
# 保存模型
# saver = tf.train.Saver({'weight': w, 'bias': b})
# with tf.Session() as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for epoch in range(epochs):
#         for (x, y) in zip(train_X, train_Y):
#             sess.run(optimizer, feed_dict={X: x, Y: y})
#         # saver.save(sess, pathname)保存模型
#         if (epoch+1) % 5 == 0:
#             saver.save(sess, "20200528/linear_model_"+str(epoch))
#             print("epoch:", epoch+1,
#                   "cost=", sess.run(cost, feed_dict={X: train_X, Y: train_Y}),
#                   "w=", sess.run(w),
#                   "b=", sess.run(b))
#     载入模型
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, '20200528/linear_model')
#     saver.save(sess, '20200528/linear_model.ckpt')


# 实例9，分析模型内容，演示模型的其他保存方法
# w = tf.Variable(1.0, name='weight')
# b = tf.Variable(5.0, name='bias')
# 保存模型的其他方法
# 1）saver = tf.train.Saver({'weight': b, 'bias': w})
# 2）saver = tf.train.Saver({v.op.name: v for v in [w, b]})
# 3) saver = tf.train.Saver([w, b])
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.save(sess, "20200528/linear_model.ckpt")
# # 查看模型里面保存了什么内容
#     from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#     print_tensors_in_checkpoint_file("20200528/linear_model.ckpt", None, True, True)


# 实例10，为模型添加检查点
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
tf.reset_default_graph()
# 搭建模型
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.get_variable("weight", shape=[1], initializer=tf.random_normal_initializer)
# w = tf.Variable(tf.random_normal([1]), name="weight")
# b = tf.Variable(tf.zeros([1]), name="bias")
b = tf.get_variable("bias", shape=[1], initializer=tf.random_normal_initializer)
# 正向
z = tf.multiply(X, w) + b
tf.summary.histogram('z', z)
# 反向
cost = tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar('loss_function', cost)
learning_rate = 0.001
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化所有变量
init = tf.global_variables_initializer()
# 定义超参数
train_epochs = 200
display_step = 2
saver = tf.train.Saver(max_to_keep=1)  # 在迭代过程中只保存三个文件
# 启动Session
with tf.Session() as sess:
    sess.run(init)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('log/', sess.graph)
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
            summary_str = sess.run(merged_summary_op, feed_dict={X: x, Y: y})
            summary_writer.add_summary(summary_str, epoch)
            saver.save(sess, 'log/linear_model.cpkt', global_step=epoch)
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
    # print("x = 0.5, y = ", sess.run(z, feed_dict={X: 0.5}))
    # 添加检查点
    # load_epoch = 18
    # with tf.Session() as sess2:
    #     sess2.run(tf.global_variables_initializer())
    #     ckpt = tf.train.get_checkpoint_state('log/')
    #     if ckpt and ckpt.model_checkpoint_path:
    #         saver.restore(sess2, ckpt.model_checkpoint_path)
    #     # saver.restore(sess2, 'log/linear_model.cpkt-' + str(load_epoch))
    #     print("x = 0.2, z = ", sess2.run(z, feed_dict={X: 0.5}))


# 实例11， 更简便地保存检查点
# tf.reset_default_graph()
# global_step = tf.train.get_or_create_global_step()
# step = tf.assign_add(global_step, 1)
# with tf.train.MonitoredTrainingSession(checkpoint_dir='log/checkpoints', save_checkpoint_secs=2) as sess:
#     print(sess.run([global_step]))
#     while not sess.should_stop():
#         i = sess.run(step)
#         print(i)


