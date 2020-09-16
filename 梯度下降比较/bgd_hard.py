"""

"""

import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib import font_manager  # 修改字体用

plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体设置
plt.rcParams['axes.unicode_minus'] = False

# Data
x = np.arange(-6 * np.pi, 90 * np.pi, 0.5)
a_real = 18.0
b_real = 0.02
c_real = 9.0
d_real = 0.05
e_real = 15.0
f_real = 0.01
y_real = a_real * np.cos(b_real * np.pi * x) + c_real * np.cos(d_real * np.pi * x) + e_real * np.cos(f_real * np.pi * x)
y_data = y_real + np.random.randn(len(x)) * 5


# plt.plot(x,y_data,'g.',markersize=2)
# plt.plot(x,y_real,'b-',markersize=2)
# plt.legend(["真实曲线","数据散点"])
# plt.title("真值")
# plt.show()
# exit()


# 梯度下降法
def gd_function(a, b, c, d, e, f, x, y_data, lr):
    m = len(x)
    grad = a * np.cos(b * np.pi * x) + c * np.cos(d * np.pi * x) + e * np.cos(f * np.pi * x) - y_data
    a_grad = 1 / m * np.sum(np.cos(b * np.pi * x) * grad)
    c_grad = 1 / m * np.sum(np.cos(d * np.pi * x) * grad)
    e_grad = 1 / m * np.sum(np.cos(f * np.pi * x) * grad)
    b_grad = 1 / m * np.sum(-a * np.sin(b * np.pi * x) * np.pi * x * grad)
    d_grad = 1 / m * np.sum(-a * np.sin(d * np.pi * x) * np.pi * x * grad)
    f_grad = 1 / m * np.sum(-a * np.sin(f * np.pi * x) * np.pi * x * grad)
    a = a - lr * a_grad
    b = b - lr * b_grad
    c = c - lr * c_grad
    d = d - lr * d_grad
    e = e - lr * e_grad
    f = f - lr * f_grad
    return a, b, c, d, e, f


# 学习率收缩
def get_learning_rate(temp, epoch, lr_sd):
    if temp <= epoch // 16:
        return lr_sd
    else:
        return lr_sd / 4


# 小批次梯度下降法
random_list = np.arange(0, len(x))
RAN_LIST = random_list.tolist()


def bgd_function(a, b, c, d, e, f, x, y_data, lr, item):
    m = len(x)
    batch = random.sample(RAN_LIST, item)
    # 这里是小批次的设计方法，首先是在外围初始化一个从1->m的全局变量
    grad = a * np.cos(b * np.pi * x[batch]) + c * np.cos(d * np.pi * x[batch]) + e * np.cos(f * np.pi * x[batch]) - \
           y_data[batch]
    a_grad = 1 / m * np.sum(np.cos(b * np.pi * x[batch]) * grad)
    c_grad = 1 / m * np.sum(np.cos(d * np.pi * x[batch]) * grad)
    e_grad = 1 / m * np.sum(np.cos(f * np.pi * x[batch]) * grad)
    b_grad = 1 / m * np.sum(-a * np.sin(b * np.pi * x[batch]) * np.pi * x[batch] * grad)
    d_grad = 1 / m * np.sum(-a * np.sin(d * np.pi * x[batch]) * np.pi * x[batch] * grad)
    f_grad = 1 / m * np.sum(-a * np.sin(f * np.pi * x[batch]) * np.pi * x[batch] * grad)
    a = a - lr * a_grad
    b = b - lr * b_grad
    c = c - lr * c_grad
    d = d - lr * d_grad
    e = e - lr * e_grad
    f = f - lr * f_grad
    return a, b, c, d, e, f


# 随机梯度下降
def sgd_function(a, b, c, d, e, f, x, y_data, lr):
    m = len(x)
    z = np.random.randint(0, m)
    grad = a * np.cos(b * np.pi * x[z]) + c * np.cos(d * np.pi * x[z]) + e * np.cos(f * np.pi * x[z]) - y_data[z]
    a_grad = 1 / m * np.sum(np.cos(b * np.pi * x[z]) * grad)
    c_grad = 1 / m * np.sum(np.cos(d * np.pi * x[z]) * grad)
    e_grad = 1 / m * np.sum(np.cos(f * np.pi * x[z]) * grad)
    b_grad = 1 / m * np.sum(-a * np.sin(b * np.pi * x[z]) * np.pi * x[z] * grad)
    d_grad = 1 / m * np.sum(-a * np.sin(d * np.pi * x[z]) * np.pi * x[z] * grad)
    f_grad = 1 / m * np.sum(-a * np.sin(f * np.pi * x[z]) * np.pi * x[z] * grad)
    a = a - lr * a_grad
    b = b - lr * b_grad
    c = c - lr * c_grad
    d = d - lr * d_grad
    e = e - lr * e_grad
    f = f - lr * f_grad
    return a, b, c, d, e, f


# 计算结果
def calculate_hat(a, b, c, d, e, f, x):
    # y_hat = a * np.exp(x) + b * x ** 3 + c * x ** 2
    y_hat = a * np.cos(b * np.pi * x) + c * np.cos(d * np.pi * x) + e * np.cos(f * np.pi * x)
    return y_hat


# 计算损失函数
def loss_function(y_data, y_hat):
    m = len(y_data)
    loss = 1 / m * np.sum((y_data - y_hat) ** 2)
    return loss


if __name__ == "__main__":
    # 超参数
    LR = 0.000002
    BATCH_SIZE = 3
    EPOCH = 60000
    # 记忆
    avg_loss = []
    # 初始化参数
    a_hat = 20 * np.random.rand(1)  # 18
    b_hat = 0.1 * np.random.rand(1)  # 0.02
    c_hat = 10 * np.random.rand(1)  # 9.0
    d_hat = 0.1 * np.random.rand(1)  # 0.05
    e_hat = 10 * np.random.rand(1)  # 15.0
    f_hat = 0.01  # * np.random.rand(1)  # 0.001
    # # 迭代
    for item in range(0, EPOCH):
        y_hat = calculate_hat(a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, x)
        a_loss = loss_function(y_data, y_hat)
        avg_loss.append(a_loss)

        if item % (EPOCH // 10) == 0:
            print("theta is:%.1f|" % a_hat, "%.3f|" % b_hat, "%.1f|" % c_hat, "%.3f|" % d_hat, "%.1f|" % e_hat,
                  "%.3f|" % f_hat, "epoch %.2f" % item, "loss %.2f|" % a_loss)

        # FIXME:如果需要使用随机梯度下降就换成sgd_function\梯度下降就是gd_function函数
        a_hat, b_hat, c_hat, d_hat, e_hat, f_hat = bgd_function(a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, x, y_data, LR,
                                                                BATCH_SIZE)
        # FIXME:这个地方要注意  这里是用的bgd代表小批次梯度下降-------------------------

    pass

    y_hat = calculate_hat(a_hat, b_hat, c_hat, d_hat, e_hat, f_hat, x)
    print("------ loss of y_now ------")
    print("theta is:%.1f|" % a_hat, "%.3f|" % b_hat, "%.1f|" % c_hat, "%.3f|" % d_hat, "%.1f|" % e_hat, "%.3f|" % f_hat,
          "loss is:%.1f" % loss_function(y_data, y_hat))
    print("------ loss of y_real ------")
    print("theta is:%.1f|" % a_real, "%.3f|" % b_real, "%.1f|" % c_real, "%.3f|" % d_real, "%.1f|" % e_real,
          "%.3f|" % f_real,
          "loss is:%.1f" % loss_function(y_data, y_real))

    # 结果 - 绘图
    plt.plot(np.arange(0, EPOCH), avg_loss, 'r-')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("损失函数随迭代次数变化")

    plt.figure()
    plt.plot(x, y_real, 'b-')
    plt.plot(x, y_data, 'g.')
    plt.plot(x, y_hat, 'r-')
    plt.title("真实曲线")
    plt.legend(["真实曲线", "数据散点", "拟合结果"])
    plt.show()
