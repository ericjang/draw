#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 14:15:42 2017

@author: zhaoxm
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

A, B, C = 100, 75, 3
eps = 1e-8

def filterbank(gx, gy, sigma2,delta, N):
    grid_i = tf.reshape(tf.cast(tf.range(N), tf.float32), [1, -1])
    mu_x = gx + (grid_i - N / 2.0 + 0.5) * delta # eq 19
    mu_y = gy + (grid_i - N / 2.0 + 0.5) * delta # eq 20
    a = tf.reshape(tf.cast(tf.range(A), tf.float32), [1, 1, -1])
    b = tf.reshape(tf.cast(tf.range(B), tf.float32), [1, 1, -1])
    mu_x = tf.reshape(mu_x, [-1, N, 1])
    mu_y = tf.reshape(mu_y, [-1, N, 1])
    sigma2 = tf.reshape(sigma2, [-1, 1, 1])
    Fx = tf.exp(-tf.square((a - mu_x)) / (2*sigma2)) # 2*sigma2?
    Fy = tf.exp(-tf.square((b - mu_y)) / (2*sigma2)) # batch x N x B
    # normalize, sum over A and B dims
    Fx=Fx/tf.maximum(tf.reduce_sum(Fx,2,keep_dims=True),eps)
    Fy=Fy/tf.maximum(tf.reduce_sum(Fy,2,keep_dims=True),eps)
    return Fx,Fy
    
img = plt.imread('./img/elephant.jpg')
img = imresize(img, [B, A])
plt.figure(1)
plt.imshow(img)
img = tf.cast(img, dtype=tf.float32)
gx, gy, sigma2, N = 75, 50, 2.0, 12 #(A-1)/2.0, (B-1)/2.0, 1.0, 5
delta = 3 #(max(A,B)-1)/(N-1)
Fx,Fy = filterbank(gx, gy, sigma2,delta, N)
Fx = tf.tile(Fx, [3, 1, 1])
Fy = tf.tile(Fy, [3, 1, 1])
Fxt=tf.transpose(Fx,perm=[0,2,1])
img=tf.transpose(img, [2, 0, 1])
glimpse = tf.matmul(Fy,tf.matmul(img,Fxt))
patch = tf.transpose(glimpse, [1, 2, 0])
Fyt = tf.transpose(Fy, [0, 2, 1])
recons = tf.matmul(Fyt,tf.matmul(glimpse,Fx))
recons = tf.transpose(recons, [1, 2, 0])

sess = tf.Session()
atte_patch, recons_img, FX, FY = sess.run([patch, recons, Fx, Fy])
FX, FY = FX[0], FY[0]
plt.figure(2)
plt.imshow(atte_patch/atte_patch.max())
plt.figure(3)
plt.imshow(recons_img/recons_img.max())
plt.figure(4)
filter = np.zeros([B, A])
for i in range(N):
    for j in range(N):
        filter += np.outer(FY[i,:].T, FX[j,:])
plt.imshow(filter/filter.max())















