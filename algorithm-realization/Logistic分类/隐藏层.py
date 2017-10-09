#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import axes3d

#
# import importlib
# importlib.import_module('mpl_toolkits.mplot3d').Axes3D

from matplotlib import cm

'''
http://www.jianshu.com/p/8e1e6c8f6d52
在这篇教程中，我们将输入数据x分类成两个类别，用蓝色表示t = 1，用红色表示t = 0。
其中，红色分类样本是一个多峰分布，被蓝色分类样本包围。这些数据都是一维的，但是数据之间的间隔并不是线性的分割。
这些数据特性将在下图中表示出来。
'''
# Define and generate the samples
nb_of_sample_per_class = 20
blue_mean = [0]
red_left_mean = [-2]
red_right_mean = [2]

std_dev = 0.5 # standard deviation of both classes
# Generate samples from both classes
x_blue = np.random.randn(nb_of_sample_per_class,1)*std_dev + blue_mean
x_red_left = np.random.randn(nb_of_sample_per_class/2,1)*std_dev + red_left_mean
x_red_right = np.random.randn(nb_of_sample_per_class/2,1)*std_dev + red_right_mean

# Merge samples in set of input variables x, and corresponding set of
x = np.vstack((x_blue,x_red_left,x_red_right))
t = np.vstack((np.ones((x_blue.shape[0],1)), np.zeros((x_red_left.shape[0],1)), np.zeros((x_red_right.shape[0],1)) ))

# Plot samples from both classes as lines on a 1D space
plt.figure(figsize=(8,0.5))
plt.xlim(-3,3)
plt.ylim(-1,1)

plt.plot(x_blue,np.zeros_like(x_blue),'b|',ms=30)
plt.plot(x_red_left, np.zeros_like(x_red_left), 'r|', ms = 30)
plt.plot(x_red_right, np.zeros_like(x_red_right), 'r|', ms = 30)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples  from the blue and red class')
plt.xlabel('$x$', fontsize=15)
plt.show()

# Define the rbf function
def rbf(z):
    return np.exp(-z**2)

# Plot the rbf function
z = np.linspace(-6,6,100)
plt.plot(z,rbf(z),'b-')
plt.xlabel('$z$', fontsize=15)
plt.ylabel('$e^{-z^2}$', fontsize=15)
plt.title('RBF function')
plt.grid()
plt.show()

'''
在计算正向传播中，输入数据被一层一层的计算，最后从模型中得出输出结果。
计算隐藏层的激活函数
计算输出结果的激活函数output_activations(h, w0)
'''
# Define the logistic function
def logistic(z):
    return 1/(1+np.exp(-z))

# Function to compute the hidden activations
def hidden_activations(x,wh):
    return rbf(x*wh)

# Define output layer feedforward
def output_activations(h,wo):
    return logistic(h*wo - 1)

# Define the neural network function
def nn(x,wh,wo):
    return output_activations(hidden_activations(x,wh),wo)

# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,wh,wo):
    return np.around(nn(x,wh,wo))

'''
在反向传播过程中，我们需要先计算出神经网络的输出与真实值之间的误差。这个误差会一层一层的反向传播去更新神经网络中的各个权重。
在每一层中，使用梯度下降算法按照负梯度方向对每个参数进行更新。
参数wh和wo利用w(k+1)=w(k)−Δw(k+1)更新，其中Δw=μ∗∂ξ/∂w，μ是学习率，∂ξ/∂w是损失函数ξ对参数w的梯度。
'''
# Define the cost function
def cost(y,t):
    return -np.sum(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))

# Define a function to calculate the cost for a given set of parameters
def cost_for_param(x,wh,wo,t):
    return cost(nn(x,wh,wo),t)

# Plot the cost in function of the weights
# Define a vector of weights for which we want to plot the cost
nb_of_ws = 200 # compute the cost nb_of_ws times in each dimension
wsh = np.linspace(-10,10,num=nb_of_ws) #hidden weights
wso = np.linspace(-10,10,num=nb_of_ws) #output weights
print(wsh)
ws_x,ws_y = np.meshgrid(wsh,wso) #generate grid
print(ws_x)
print(ws_x.shape)
cost_ws = np.zeros((nb_of_ws, nb_of_ws)) # initialize cost matrix

# Fill the cost matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i,j] = cost(nn(x,ws_x[i,j],ws_y[i,j]),t)

# Plot the cost function surface
fig = plt.figure()
ax = Axes3D(fig)
# plot the surface
surf = ax.plot_surface(ws_x,ws_y,cost_ws,linewidth=0,cmap=cm.pink)
ax.view_init(elev=60,azim=-30) #设置初始视图来改变角度
cbar = fig.colorbar(surf)
ax.set_xlabel('$w_h$', fontsize=15)
ax.set_ylabel('$w_o$', fontsize=15)
ax.set_zlabel('$\\xi$', fontsize=15)
cbar.ax.set_ylabel('$\\xi$', fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()

'''
gradient_output(y, t)函数实现了δo
gradient_weight_out(h, grad_output)函数实现了∂ξ/∂wo。
gradient_hidden(wo, grad_output)函数实现了δh。
gradient_weight_hidden(x, zh, h, grad_hidden)函数实现了∂ξ/∂wh。
backprop_update(x, t, wh, wo, learning_rate)函数实现了BP算法的每次迭代过程。
'''
# Define the error function
def gradient_output(y,t):
    return y-t

# Define the gradient function for the weight parameter at the output layer
def gradient_weight_out(h,grad_output):
    return h*grad_output

# Define the gradient function for the hidden layer
def gradient_hidden(wo,grad_output):
    return wo*grad_output

# Define the gradient function for the weight parameter at the hidden layer
def gradient_weight_hidden(x,zh,h,grad_hidden):
    return x * -2 *zh*h*grad_hidden

# Define the update function to update the network parameters over 1 iteration
def backprop_update(x,t,wh,wo,learning_rate):
    # Compute the output of the network
    # This can be done with y = nn(x, wh, wo), but we need the intermediate
    #  h and zh for the weight updates.
    zh = x*wh
    h = rbf(zh)  # hidden_activations(x, wh)
    y = output_activations(h,wo)
    # Compute the gradient at the output
    grad_output = gradient_output(y,t)
    # Get the delta for wo
    d_wo = learning_rate*gradient_weight_out(h,grad_output)

    # Compute the gradient at the hidden layer
    grad_hidden = gradient_hidden(wo,grad_output)
    # Get the delta for wh
    d_wh = learning_rate*gradient_weight_hidden(x,zh,h,grad_hidden)
    # return the update parameters
    return (wh-d_wh.sum(),wo-d_wo.sum())

'''
BP算法更新

下面的代码，我们模拟了一个50次的循环。白色的点表示，参数wh和wo在误差面上面的第k次迭代。

在更新过程中，我们不断的线性减小学习率。这是为了在更新到最后的时候，学习率能是0。这样能保证最后的参数更新不会在最小值附近徘徊。
'''
# Run backpropagation
# Set the initial weight parameterv
wh = 2
wo = -5
learning_rate = 0.2

# Start the gradient descent updates and plot the iterations
nb_of_iterations = 50
lr_update = learning_rate / nb_of_iterations #learning rate update rule
w_cost_iter = [(wh,wo,cost_for_param(x,wh,wo,t))] # List to store the weight values over the iterations
for i in range(nb_of_iterations):
    learning_rate -= lr_update # decrease the learning rate
    # Update the weights via backpropagation
    wh,wo = backprop_update(x,t,wh,wo,learning_rate)
    w_cost_iter.append((wh,wo,cost_for_param(x,wh,wo,t)))

# Print the final cost
print('final cost is {:.2f} for weights wh: {:.2f} and wo: {:.2f}'.format(cost_for_param(x,wh,wo,t),wh,wo))

#画出梯度下降的过程
fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(ws_x,ws_y,cost_ws,linewidth=0,cmap=cm.pink)
ax.view_init(elev=60,azim=-30)
cbar = fig.colorbar(surf)
cbar.ax.set_ylabel('$\\xi$',fontsize=15)

#plot the updates
for i in range(1,len(w_cost_iter)):
    wh1,wo1,c1 = w_cost_iter[i-1]
    wh2,wo2,c2 = w_cost_iter[i]
    ax.plot([wh1],[wo1],[c1],'w+')
    ax.plot([wh1,wh2],[wo1,wo2],[c1,c2],'w-')
# Plot the last weights
wh1,wo1,c1 = w_cost_iter[len(w_cost_iter)-1]
ax.plot([wh1],[wo1],c1,'w+')
ax.set_xlabel('$w_h$',fontsize=15)
ax.set_ylabel('$w_o$',fontsize=15)
ax.set_zlabel('$\\xi$',fontsize=15)
plt.title('Gradient descent updates on cost surface')
plt.grid()
plt.show()

'''
分类结果的可视化
下面的代码可视化了最后的分类结果。在输入空间域里面，蓝色和红色代表了最后的分类颜色。从图中，我们发现所有的样本都被正确分类了。
'''
nb_of_xs = 100
xs = np.linspace(-3,3,num=nb_of_xs)
ys = np.linspace(-1,1,num=nb_of_xs)
xx,yy = np.meshgrid(xs,ys)
# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs,nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i][j] = nn_predict(xx[i,j],wh,wo)

# Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
    colorConverter.to_rgba('r',alpha=0.25),
    colorConverter.to_rgba('b',alpha=0.25)
])

plt.figure(figsize=(8,0.5))
plt.contourf(xx,yy,classification_plane,cmap=cmap)
plt.xlim(-3,3)
plt.ylim(-1,1)
# Plot samples from both classes as lines on a 1D space
plt.plot(x_blue,np.zeros_like(x_blue),'b|',ms = 30)
plt.plot(x_red_left,np.zeros_like(x_red_left),'r|',ms = 30)
plt.plot(x_red_right,np.zeros_like(x_red_right),'r|',ms = 30)
plt.gca().axes.get_yaxis().set_visible(False)
plt.title('Input samples and their classification')
plt.xlabel('x')
plt.show()
