#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, ListedColormap
from matplotlib import cm
'''
在教程中，目标分类t将从两个独立分布中产生，当t=1时，用蓝色表示。当t=0时，用红色表示。输入参数X是一个N*2的矩阵，目标分类t是一个N * 1的向量。
'''
# Define and generate the samples
nb_of_sample_per_class = 20 #the number of red class
red_mean = [-1,0]
blue_mean = [-1,0]

std_dev = 1.2  # standard deviation of both classes

# Generate samples from both classes
x_red = np.random.randn(nb_of_sample_per_class,2)*std_dev + red_mean
x_blue = np.random.randn(nb_of_sample_per_class,2)*std_dev + blue_mean

# Merge samples in set of input variables x, and corresponding set of output variables t
X = np.vstack((x_red,x_blue))
t = np.vstack((np.zeros((nb_of_sample_per_class,1)),np.ones((nb_of_sample_per_class,1))))

# Plot both classes on the x1, x2 plane
plt.plot(x_red[:,0],x_red[:,1],'ro',label='class red')
plt.plot(x_blue[:,0],x_blue[:,1],'bo',label='class blue')
plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.axis([-4, 4, -4, 4])
plt.title('red vs. blue classes in the input space')
plt.show()

'''
logistic(z)函数实现了Logistic函数，cost(y, t)函数实现了损失函数，nn(x, w)实现了神经网络的输出结果，
nn_predict(x, w)实现了神经网络的预测结果。
'''
# Define the logistic function
def logistic(z):
    return 1/(1+np.exp(-z))

# Define the neural network function y = 1 / (1 + numpy.exp(-x*w))
def nn(x,w):
    return logistic(x.dot(w.T))

# Define the neural network prediction function that only returns
#  1 or 0 depending on the predicted class
def nn_predict(x,w):
    return np.around(nn(x,w))

# Define the cost function
def cost(y,t):
    return - np.sum(np.multiply(t,np.log(y)) + np.multiply((1-t),np.log(1-y)))

# Plot the cost in function of the weights
# Define a vector of weights for which we want to plot the cost
nb_of_ws = 100  # compute the cost nb_of_ws times in each dimension
ws1 = np.linspace(-5,5,num=nb_of_ws) # weight 1
ws2 = np.linspace(-5,5,num=nb_of_ws)
ws_x,ws_y = np.meshgrid(ws1,ws2) # generate grid
cost_ws = np.zeros((nb_of_ws,nb_of_ws))  # initialize cost matrix

# Fill the cost matrix for each combination of weights
for i in range(nb_of_ws):
    for j in range(nb_of_ws):
        cost_ws[i,j] = cost(nn(X,np.asmatrix([ws_x[i,j],ws_y[i,j]])),t)

# Plot the cost function surface
plt.contourf(ws_x,ws_y,cost_ws,20,cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('$\\xi$', fontsize=15)
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Cost function surface')
plt.grid()
plt.show()

'''
梯度下降算法的工作原理是损失函数ξ对于每一个参数的求导，然后沿着负梯度方向进行参数更新。
gradient(w, x, t)函数实现了梯度∂ξ/∂w，delta_w(w_k, x, t, learning_rate)函数实现了Δw。
'''
# define the gradient function.
def gradient(w,x,t):
    return(nn(x,w)-t).T*x

# define the update function delta w which returns the
#  delta w for each weight in a vector
def delta_w(w_k,x,t,learning_rate):
    return learning_rate*gradient(w_k,x,t)

'''
我们在训练集X上面运行10次去做预测，下图中画出了前三次的结果，图中蓝色的点表示在第k次，w(k)的值。
'''
# Set the initial weight parameter
w = np.asmatrix([-4,-2])
print w
# Set the learning rate
learning_rate = 0.05

# Start the gradient descent updates and plot the iterations
nb_of_iterations = 10 # Number of gradient descent updates
w_iter = [w] # List to store the weight values over the iterations
for  i in range(nb_of_iterations):
    dw = delta_w(w,X,t,learning_rate)  # Get the delta w update
    w = w - dw
    w_iter.append(w)  # Store the weights for plotting

# Plot the first weight updates on the error surface
# Plot the error surface
plt.contourf(ws_x,ws_y,cost_ws,20,alpha=0.9,cmap=cm.pink)
cbar = plt.colorbar()
cbar.ax.set_ylabel('cost')

# Plot the updates
for i in range(1,4):
    w1 = w_iter[i-1]
    w2 = w_iter[i]
    # Plot the weight-cost value and the line that represents the update
    plt.plot(w1[0,0],w1[0,1],'bo')  # Plot the weight cost value
    plt.plot([w1[0,0],w2[0,0]],[w1[0,1],w2[0,1]],'b-')
    plt.text(w1[0,0]-0.2,w[0,1]+0.4,'$w({})$'.format(4),color='b')

# Show figure
plt.xlabel('$w_1$', fontsize=15)
plt.ylabel('$w_2$', fontsize=15)
plt.title('Gradient descent updates on cost surface')
plt.grid()
plt.show()

'''
训练结果可视化
'''
# Plot the resulting decision boundary
# Generate a grid over the input space to plot the color of the
#  classification at that grid point
nb_of_xs = 200
xs1 = np.linspace(-4,4,num=nb_of_xs)
xs2 = np.linspace(-4,4,num=nb_of_xs)
xx,yy = np.meshgrid(xs1,xs2)

# Initialize and fill the classification plane
classification_plane = np.zeros((nb_of_xs,nb_of_xs))
for i in range(nb_of_xs):
    for j in range(nb_of_xs):
        classification_plane[i,j] = nn_predict(np.asmatrix([xx[i,j],yy[i,j]]),w)

## Create a color map to show the classification colors of each grid point
cmap = ListedColormap([
    colorConverter.to_rgba('r',alpha=0.30),
    colorConverter.to_rgba('b',alpha=0.30)
])

# Plot the classification plane with decision boundary and input samples
plt.contourf(xx,yy,classification_plane,cmap=cmap)
plt.plot(x_red[:,0],x_red[:,1],'ro',label='target red')
plt.plot(x_blue[:,0],x_blue[:,1],'bo',label='target blue')

plt.grid()
plt.legend(loc=2)
plt.xlabel('$x_1$', fontsize=15)
plt.ylabel('$x_2$', fontsize=15)
plt.title('red vs. blue classification boundary')
plt.show()
