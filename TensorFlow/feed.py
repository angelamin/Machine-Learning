#!/usr/bin/python
# -*- coding: UTF-8 -*-
import tensorflow as tf

input1 = tf.placeholder(tf.types.float32) #提供占位符
input2 = tf.placeholder(tf.types.float32)
output = tf.mul(input1,input2)

with tf.Session() as sess:
    print sess.run([output],feed_dict={input1:[7.],input2:[2.]})
# 输出:
# [array([ 14.], dtype=float32)]
