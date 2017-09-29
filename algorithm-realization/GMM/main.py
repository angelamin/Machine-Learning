#! /usr/bin/env python
#coding=utf-8

import GMM
'''
def showFigure(dataMat,k,clusterAssment):

    tag=['go','or','yo','ko']
    for i in range(k):

        datalist = dataMat[nonzero(clusterAssment[:,0].A==i)[0]]
        pylab.plot(datalist[:,0],datalist[:,1],tag[i])
    pylab.show()
'''
if __name__ == '__main__':
    GMM.gmm('testSet.txt',2)
