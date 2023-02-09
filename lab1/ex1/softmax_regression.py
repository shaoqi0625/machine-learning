# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


# softmax_regression
def softmax_regression(theta, x, y, iters, alpha):
    # TODO: Do the softmax regression by computing the gradient and
    # the objective function value of every iteration and update the theta
    f = list()  # 损失函数
    lam = 0.01

    for i in range(iters):
        # 计算 m * k 的分数矩阵
        scores = np.dot(theta, x.T)
        # 计算softmax值
        sum_exp = np.sum(np.exp(scores), axis=0)
        softmax = np.exp(scores) / sum_exp
        # 计算损失函数值
        loss = 0.0
        softmax_log = np.log(softmax)
        for i in range(len(x)):
            loss += np.dot(softmax_log[:, i].T, y[:, i])
        loss = - (1.0 / len(x)) * loss
        loss = loss + lam * np.sum(theta ** 2)
        f.append(loss)
        # 求解梯度
        g = -(1.0 / len(x)) * np.dot((y - softmax), x) + lam * theta
        # 更新权重矩阵
        theta = theta - alpha * g

    fig = plt.figure(figsize=(8, 5))
    plt.plot(np.arange(iters), f)
    plt.title("Development of loss during training")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.show()
    return theta
