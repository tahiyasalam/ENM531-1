#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 13:20:11 2018

@author: Paris
"""

import tensorflow as tf
import numpy as np
import timeit


class RNN:
    # Initialize the class
    def __init__(self, X, Y, hidden_dim):
        
        # X has the form lags x data x dim
        # Y has the form data x dim
     
        self.X = X
        self.Y = Y
        
        self.X_dim = X.shape[-1]
        self.Y_dim = Y.shape[-1]
        self.hidden_dim = hidden_dim
        self.lags = X.shape[0]

        # Initialize network weights and biases        
        self.U, self.b, self.W, self.V, self.c = self.initialize_RNN()
                
        # Store loss values
        self.training_loss = [] 
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_tf = tf.placeholder(tf.float32, shape=(self.X.shape[0], None, self.X.shape[2]))
        self.Y_tf = tf.placeholder(tf.float32, shape=(None, self.Y.shape[1]))
        
        # Evaluate prediction
        self.Y_pred = self.forward_pass(self.X_tf)
        
        # Evaluate loss
        self.loss = tf.losses.mean_squared_error(self.Y_tf, self.Y_pred)
        
        # Define optimizer        
        self.optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = self.optimizer.minimize(self.loss)
        
        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_RNN(self):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)    
        
        U = xavier_init(size=[self.X_dim, self.hidden_dim])
        b = tf.Variable(tf.zeros([1,self.hidden_dim], dtype=tf.float32), dtype=tf.float32)
            
        W = tf.Variable(tf.eye(self.hidden_dim, dtype=tf.float32), dtype=tf.float32)
            
        V = xavier_init(size=[self.hidden_dim, self.Y_dim])
        c = tf.Variable(tf.zeros([1,self.Y_dim], dtype=tf.float32), dtype=tf.float32)
            
        return U, b, W, V, c
    
           
    # Evaluates the forward pass
    def forward_pass(self, X):
        H = tf.zeros([tf.shape(X)[1], self.hidden_dim], dtype=tf.float32)
        for i in range(0, self.lags):
            H = tf.nn.tanh(tf.matmul(H,self.W) + tf.matmul(X[i,:,:],self.U) + self.b)       
        H = tf.matmul(H,self.V) + self.c
        return H
    
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self,X, Y, N_batch):
        N = X.shape[1]
        idx = np.random.choice(N, N_batch, replace=False)
        X_batch = X[:,idx,:]
        Y_batch = Y[idx,:]        
        return X_batch, Y_batch
    
    
    # Trains the model by minimizing the MSE loss
    def train(self, nIter = 10000, batch_size = 100): 

        start_time = timeit.default_timer()
        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_batch, Y_batch = self.fetch_minibatch(self.X, self.Y, batch_size)
            
            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_tf: X_batch, self.Y_tf: Y_batch}  
            
            # Run the Tensorflow session to minimize the loss
            self.sess.run(self.train_op, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = timeit.default_timer()
                
                
    # Evaluates predictions at test points           
    def predict(self, X_star):      
        tf_dict = {self.X_tf: X_star}       
        Y_star = self.sess.run(self.Y_pred, tf_dict) 
        return Y_star
    
    
