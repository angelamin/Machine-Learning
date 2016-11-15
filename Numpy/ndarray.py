
import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt("/Users/xiamin/Downloads/ML_hit/Task1/q2x.dat",dtype=float).ravel()
y = np.loadtxt("/Users/xiamin/Downloads/ML_hit/Task1/q2y.dat",dtype=float).ravel()

plt.plot(x,y,'g^',zorder=20)
plt.xlabel('x')
plt.ylabel('y')

theta0 = 0.
theta1 = 1.

a = 0.00001
xlen = float(len(x))

while True:
    sum0 = 0
    sum1 = 0
    temp = 0
    for i in range(len(x)):
        temp = theta0 + theta1*x[i] - y[i]
        sum0 += temp
        sum1 += temp*x[i]
    temp0 = theta0 - a*1./xlen*sum0
    temp1 = theta1 - a*(1./xlen)*sum1
    if abs(theta0 - temp0) <= 0.00001 and abs(theta1 - temp1) <= 0.00001:
        theta0 = temp0
        theta1 = temp1
        break
    theta0 = temp0
    theta1 = temp1
    plt.plot(x,theta0 + theta1*x,c=np.random.rand(3,1),zorder=10)

plt.show()