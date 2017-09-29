#! /usr/bin/env python
#coding=utf-8

from numpy import *
import pylab
import random,math

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def gmm(file, K_or_centroids):
# ============================================================
# Expectation-Maximization iteration implementation of
# Gaussian Mixture Model.
#
# PX = GMM(X, K_OR_CENTROIDS)
# [PX MODEL] = GMM(X, K_OR_CENTROIDS)
#
#  - X: N-by-D data matrix.
#  - K_OR_CENTROIDS: either K indicating the number of
#       components or a K-by-D matrix indicating the
#       choosing of the initial K centroids.
#
#  - PX: N-by-K matrix indicating the probability of each
#       component generating each point.
#  - MODEL: a structure containing the parameters for a GMM:
#       MODEL.Miu: a K-by-D matrix.
#       MODEL.Sigma: a D-by-D-by-K matrix.
#       MODEL.Pi: a 1-by-K vector.
# ============================================================
    ## Generate Initial Centroids
    threshold = 1e-15
    dataMat = mat(loadDataSet(file))
    [N, D] = shape(dataMat)
    K_or_centroids = 2
    # K_or_centroids可以是一个整数，也可以是k个质心的二维列向量
    if shape(K_or_centroids)==(): #if K_or_centroid is a 1*1 number
        K = K_or_centroids
        Rn_index = range(N)
        random.shuffle(Rn_index) #random index N samples
        centroids = dataMat[Rn_index[0:K], :]; #generate K random centroid
    else: # K_or_centroid is a initial K centroid
        K = size(K_or_centroids)[0];
        centroids = K_or_centroids;

    ## initial values
    [pMiu,pPi,pSigma] = init_params(dataMat,centroids,K,N,D)
    Lprev = -inf #上一次聚类的误差

    # EM Algorithm
    while True:
        # Estimation Step
        Px = calc_prob(pMiu,pSigma,dataMat,K,N,D)

        # new value for pGamma(N*k), pGamma(i,k) = Xi由第k个Gaussian生成的概率
        # 或者说xi中有pGamma(i,k)是由第k个Gaussian生成的
        pGamma = mat(array(Px) * array(tile(pPi, (N, 1))))  #分子 = pi(k) * N(xi | pMiu(k), pSigma(k))
        pGamma = pGamma / tile(sum(pGamma, 1), (1, K)) #分母 = pi(j) * N(xi | pMiu(j), pSigma(j))对所有j求和

        ## Maximization Step - through Maximize likelihood Estimation
        #print 'dtypeddddddddd:',pGamma.dtype
        Nk = sum(pGamma, 0) #Nk(1*k) = 第k个高斯生成每个样本的概率的和，所有Nk的总和为N。

        # update pMiu
        pMiu = mat(diag((1/Nk).tolist()[0])) * (pGamma.T) * dataMat #update pMiu through MLE(通过令导数 = 0得到)
        pPi = Nk/N

        # update k个 pSigma
        print 'kk=',K
        for kk in range(K):
            Xshift = dataMat-tile(pMiu[kk], (N, 1))

            Xshift.T * mat(diag(pGamma[:, kk].T.tolist()[0])) *  Xshift / 2

            pSigma[:, :, kk] = (Xshift.T * \
                mat(diag(pGamma[:, kk].T.tolist()[0])) * Xshift) / Nk[kk]

        # check for convergence
        L = sum(log(Px*(pPi.T)))
        if L-Lprev < threshold:
            break
        Lprev = L

    return Px


def init_params(X,centroids,K,N,D):
    pMiu = centroids #k*D, 即k类的中心点
    pPi = zeros([1, K]) #k类GMM所占权重（influence factor）
    pSigma = zeros([D, D, K]) #k类GMM的协方差矩阵，每个是D*D的

    # 距离矩阵，计算N*K的矩阵（x-pMiu）^2 = x^2+pMiu^2-2*x*Miu
    #x^2, N*1的矩阵replicateK列\#pMiu^2，1*K的矩阵replicateN行
    distmat = tile(sum(power(X,2), 1),(1, K)) + \
        tile(transpose(sum(power(pMiu,2), 1)),(N, 1)) -  \
        2*X*transpose(pMiu)
    labels = distmat.argmin(1) #Return the minimum from each row

    # 获取k类的pPi和协方差矩阵
    for k in range(K):
        boolList = (labels==k).tolist()
        indexList = [boolList.index(i) for i in boolList if i==[True]]
        Xk = X[indexList, :]
        #print cov(Xk)
        # 也可以用shape(XK)[0]
        pPi[0][k] = float(size(Xk, 0))/N
        pSigma[:, :, k] = cov(transpose(Xk))

    return pMiu,pPi,pSigma

# 计算每个数据由第k类生成的概率矩阵Px
def calc_prob(pMiu,pSigma,X,K,N,D):
    # Gaussian posterior probability
    # N(x|pMiu,pSigma) = 1/((2pi)^(D/2))*(1/(abs(sigma))^0.5)*exp(-1/2*(x-pMiu)'pSigma^(-1)*(x-pMiu))
    Px = mat(zeros([N, K]))
    for k in range(K):
        Xshift = X-tile(pMiu[k, :],(N, 1)) #X-pMiu
        #inv_pSigma = mat(pSigma[:, :, k]).I
        inv_pSigma = linalg.pinv(mat(pSigma[:, :, k]))

        tmp = sum(array((Xshift*inv_pSigma)) * array(Xshift), 1) # 这里应变为一列数
        tmp = mat(tmp).T
        #print linalg.det(inv_pSigma),'54545'

        Sigema = linalg.det(mat(inv_pSigma))

        if Sigema < 0:
            Sigema=0

        coef = power((2*(math.pi)),(-D/2)) * sqrt(Sigema)
        Px[:, k] = coef * exp(-0.5*tmp)
    return Px
