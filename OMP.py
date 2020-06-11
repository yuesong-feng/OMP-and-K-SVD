import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import random
import cv2
import os
from scipy import sparse
np.set_printoptions(threshold=1000)
def OMP(D,Y,T):
    if len(D.shape) > 1:
        K = D.shape[1]
    else:
        K = 1
        D = D.reshape((D.shape[0],1))
    if len(Y.shape) > 1:
        N = Y.shape[1]
    else:
        N = 1
        Y = Y.reshape((Y.shape[0],1))
    X = np.zeros((K,N))
    for i in range(N):
        y = Y[:,i]
        r = y
        indx = []
        for k in range(T):
            proj = np.fabs(np.dot(D.T,r))
            pos = np.argmax(proj)
            indx.append(pos)
            if k == 0:
                A = D[:,pos].reshape(Y.shape[0],1)
            else:
                A = np.concatenate((A,D[:,pos].reshape(Y.shape[0],1)),axis = 1)
            x = np.dot(np.linalg.pinv(A),y)
            r = y - np.dot(A,x)
        tmp = np.zeros((K,1))
        tmp[indx] = x.reshape((T,1))
        tmp = np.array(tmp).reshape(K)
        X[:,i] = tmp
    return X
def K_SVD(img,iter_times,K, T,err=1e-6): 
    Y = fenge(img) 
    n = 64
    N = Y.shape[1]
    X = np.zeros((K,N))
    D = np.random.random((n,K))
    for i in range(K):
        norm = np.linalg.norm(D[:,i])
        mean=np.sum(D[:,i])/D.shape[0]
        D[:, i] = (D[:, i]-mean) / norm 
    for i in range(N):
        norm = np.linalg.norm(Y[:,i])
        mean = np.sum(Y[:,i]) / Y.shape[0]
        Y[:,i] = (Y[:,i] - mean) / norm
    for j in range(iter_times):
        X = OMP(D,Y,T)
        e = np.linalg.norm(Y- np.dot(D, X))
        print(str('第%s次迭代，误差为：%s' %(j, e))+'\n')
        if e < err:
            break
        for k in range(K):
            index = np.nonzero(X[k, :])[0]
            if len(index) == 0:
                continue
            D[:, k] = 0
            R = (Y - np.dot(D, X))[:, index]
            u, s, v = np.linalg.svd(R, full_matrices=False)
            D[:, k] = u[:, 0].T
            X[k, index ] = s[0] * v[0, :]
    return D
def fenge(img):
    dim_r = img.shape[0] // 8
    dim_c = img.shape[1] // 8
    dim = dim_r * dim_c
    patchs = np.zeros((64, dim))
    for i in range(dim_r):
        for j in range(dim_c):
            r = i * 8
            c = j * 8
            patch = img[r:r+8,c:c+  8].reshape(64)
            patchs[:,i*dim_c + j] = patch
    return patchs
def patch_merg(patchs, shp):
    img = np.zeros(shp)
    dim_r = img.shape[0] // 8
    dim_c = img.shape[1] // 8
    for i in range(dim_r):
        for j in range(dim_c):
            r = i * 8
            c = j * 8
            img[r:r+8,c:c+8] = patchs[:,i*dim_c+j].reshape(8,8)
    return img 
def reconstruct(img, D, K):
    patchs = fenge(img)
    for i in range(patchs.shape[1]):
        patch = patchs[:,i]
        index = np.nonzero(patch)[0]
        if index.shape[0] == 0:
            continue
        l2norm=np.linalg.norm(patch[index])
        mean=np.sum(patch)/index.shape[0]
        patch_norm=(patch-mean)/l2norm
        x = OMP(D[index, :], patch_norm[index].T, K)
        patchs[:, i]=np.fabs(((D.dot(x)*l2norm)+mean).reshape(patchs.shape[0]))
    return patch_merg(patchs,img.shape)
def miss_pic(img,k = 50):
    patchs = fenge(img)
    k = int(k*0.01*patchs.shape[0]*patchs.shape[1])
    loss_r = np.random.randint(0, high = patchs.shape[0]-1,size = k)
    loss_c = np.random.randint(0, high = patchs.shape[1]-1,size = k)
    for i in range(k):
        patchs[loss_r[i],loss_c[i]] = 0
    return patchs
picture = []
for root,dir,files in os.walk('./picture/DataSet/'):
    for file in files:
        picture.append(cv2.imread('./picture/DataSet/' + str(file),-1))
train = []
test = []
for i in range(38):
    if i < 30:
        train.append(picture[i])
    else:
        test.append(picture[i])
N = 504
K = 256
T = 50
chushi = np.array(fenge(train[0]))
for i in range(1,len(train)):
    patchs = fenge(train[i])
    chushi = np.concatenate((chushi,patchs),axis = 1)
train = chushi[:,np.random.randint(0, high = chushi.shape[1]-1, size = 504)]
D = K_SVD(train,5,K,T)
for i in range(len(test)):
    loss = patch_merg(miss_pic(test[i],90),test[i].shape)
    cv2.imwrite("./picture/LossPic/loss"+str(i)+".jpg",loss.astype(np.uint8))
    print("Loss "+str(i)+" has been loaded..")
    rec_img = reconstruct(loss,D,K)
    cv2.imwrite("./picture/RecPic/rec"+str(i)+".jpg",rec_img.astype(np.uint8))
    print("Loss "+str(i)+" is reconstructed!")
