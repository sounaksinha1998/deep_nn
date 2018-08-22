import numpy as np
import tensorflow as tf

class Neural_net:
    def __init__(self,input_data,output_data,topology,learning_rate,epochs):
        self.input_data = input_data
        self.output_data = output_data
        self.topology = topology
        self.lr = learning_rate
        self.epochs = epochs

    def model(self):
        X = tf.placeholder(tf.float32)
        Y = tf.placeholder(tf.float32,shape=(np.size(self.output_data),self.topology[np.size(self.topology) -1]))
        w = []
        a = []
        for i in range(np.size(self.topology) -1):
            w.append(tf.Variable(tf.random_uniform([self.topology[i],self.topology[i+1]],-1,1)))
            if i == 0:
                a.append(tf.nn.relu(tf.matmul(X,w[i])))
            else: a.append(tf.nn.relu(tf.matmul(a[i-1],w[i])))

        cost = tf.reduce_sum(tf.square(Y - a[np.size(self.topology) -2]))


        sess = tf.InteractiveSession()
        writer = tf.summary.FileWriter('graph', sess.graph)
        optimizer = tf.train.AdamOptimizer(0.001,0.9,0.999,1e-8,False,"Adam")
        train = optimizer.minimize(cost)
        init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
        sess.run(init)

        for steps in range(self.epochs):
            sess.run(train,{X:self.input_data, Y:self.output_data})
            if steps %5000 == 0:
                print(sess.run(cost,{X:self.input_data, Y:self.output_data}))

        print(sess.run(a[np.size(self.topology) -2],{X:self.input_data, Y:self.output_data}))
        print(sess.run(a[np.size(self.topology) -2],{X: test_data}))
        writer.close()
        sess.close()

input_data = np.array([[0,0],[1,1],[1,0],[0,1]],np.float32)
test_data = np.array([[1,1],[0,0]],np.float32)
output_data = np.array([[0],[1],[1],[1]],np.float32)
learning_rate = 0.01
epochs = 50000
topology = np.array([2,3,3,1],np.int32)

net = Neural_net(input_data,output_data,topology,learning_rate,epochs)
net.model()

