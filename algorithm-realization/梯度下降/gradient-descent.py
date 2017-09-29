#coding:utf-8
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
'''
在这个例子中，我们使用函数f来产生目标结果t，但是对目标结果加上一些高斯噪声N(0, 0.2)，其中N表示正态分布，均值是0，方差是0.2，f定义为f(x) = 2x，x是输入参数，回归线的斜率是2，截距是0。
所以最后的t = f(x) + N(0, 0.2)。

我们将产生20个均匀分布的数据作为数据样本x，然后设计目标结果t。下面的程序我们生成了x和t，以及画出了他们之间的线性关系。
'''
# Define the vector of input samples as x, with 20 values sampled from a uniform distribution
# between 0 and 1
x = np.random.uniform(0,1,20)

# Generate the target values t from x with small gaussian noise so the estimation won't be perfect.
# Define a function f that represents the line that generates t without noise
def f(x):
    return x*2
# Create the targets t with some gaussian noise
noise_variance = 0.2  #噪声方差
# Gaussian noise error for each sample in x
print(x.shape[0])  #行数20
noise = np.random.randn(x.shape[0])*noise_variance
# Create targets t
t = f(x) + noise

# Plot the target t versus the input x
plt.plot(x,t,'o',label='t') #散点图
# Plot the initial line蓝色直线画
plt.plot([0,1],[f(0),f(1)],'b-',label="f(x)")
plt.xlabel('$x$',fontsize=15)
plt.ylabel('$y$',fontsize=15)
plt.ylim([0,2]) #y范围
plt.title('inputs (x) vs targets (t)')
plt.grid()
plt.legend(loc=2) #左上角变量的表示
plt.show()

'''
首先我们来构建一个最简单的神经网络，这个神经网络只有一个输入，一个输出，用来构建一个线性回归模型，从输入的x来预测一个真实结果t。
神经网络的模型结构为y = x * w，其中x是输入参数，w是权重，y是预测结果。
nn(x, w)函数实现了神经网络模型，cost(y, t)函数实现了损失函数。
'''
# Define the neural network function y = x * w
def nn(x,w):
    return x*w
# Define the cost function
def cost(y,t):
    return ((t-y)**2).sum()

'''
gradient(w, x, t)函数实现了梯度∂ξ/∂w，delta_w(w_k, x, t, learning_rate)函数实现了Δw
'''

# define the gradient function. Remember that y = nn(x, w) = x * w
# 损失函数对w求导
# 具体参见 http://www.jianshu.com/p/0da9eb3fd06b
def gradient(w,x,t):
    return 2*x*(nn(x,w)-t)

# define the update function delta w
def delta_w(w_k,x,t,learning_rate):
    return learning_rate*gradient(w_k,x,t).sum()

# Set the initial weight parameter
w = 0.1
learning_rate = 0.1

# Start performing the gradient descent updates, and print the weights and cost:
nb_of_iteration = 10  #number of gradient descent updates
w_cost = [(w,cost(nn(x,w),t))]  # List to store the weight, costs values

for i in range(nb_of_iteration):
    dw = delta_w(w,x,t,learning_rate)     # Get the delta w update
    w = w-dw                              # Update the current weight parameter
    w_cost.append((w,cost(nn(x,w),t)))     # Add weight, cost to list

# Print the final w, and cost
for i in range(0,len(w_cost)):
    print('w({}):{:.4f} \t cost:{:.4f}'.format(i,w_cost[i][0],w_cost[i][1]))

# Plot the first 2 gradient descent updates
# plt.plot(ws,cost_ws,'r-')  # Plot the error curve

'''
图展示了梯度下降的可视化过程。图中蓝色的点表示在第k轮中w(k)的值。
从图中我们可以得知，w的值越来越收敛于2.0
'''
# Plot the updates
for i in range(0,len(w_cost)-2):
    w1,c1 = w_cost[i]
    w2,c2 = w_cost[i+1]
    plt.plot(w1,c1,'bo')
    plt.plot([w1,w2],[c1,c2],'b-')
    plt.text(w1,c1+0.5,'$w({}$)'.format(i))
# Show figure
plt.xlabel('$w$',fontsize=15)
plt.ylabel('$\\xi$',fontsize=15)
plt.title('Gradient descent updates plotted on cost function')
plt.grid()
plt.show()

# Plot the fitted line agains the target line
# Plot the target t versus the input x
plt.plot(x, t, 'o', label='t')
# Plot the initial line
plt.plot([0, 1], [f(0), f(1)], 'b-', label='f(x)')
# plot the fitted line
plt.plot([0, 1], [0*w, 1*w], 'r-', label='fitted line')
plt.xlabel('input x')
plt.ylabel('target t')
plt.ylim([0,2])
plt.title('input vs. target')
plt.grid()
plt.legend(loc=2)
plt.show()
