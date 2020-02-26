import sys
import numpy as np 
import scipy
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
#import tensorflow as tf2
import tqdm
import time
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
tf.disable_v2_behavior()
def sinwave(freq, phase, sampling_freq):
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = (np.mod(phase_series,1)<0.5).astype(np.float64)*2-1
    return sequence

def sqwave(freq, phase, sampling_freq):
    phase_series = np.linspace(0, 1, sampling_freq) * freq + phase/2/np.pi
    sequence = (np.mod(phase_series,1)<0.5).astype(np.float64)*2-1
    return sequence

def sawwave(freq, phase, sampling_freq):
    phase_series = np.linspace(0, 1, sampling_freq) * freq + phase/2/np.pi
    sequence = np.mod((phase_series*2+1.),2.)-1.
    return sequence

def triwave(freq,phase,sampling_freq):
    phase_series = np.linspace(0,1,sampling_freq) * freq + phase/2/np.pi
    sequence = -np.abs(np.mod(phase_series+0.25,1.)-0.5)*4+1
    return sequence

def moving_noise(force,damp,cov,sampling_freq,seed=-1):
    if seed>0:
        np.random.seed(seed)
    x = 0.
    v = np.random.rand()
    sequence = [x]
    for _ in range(sampling_freq-1):
        v += np.random.normal()*force - cov*v - damp * x
        x += v
        sequence.append(sequence[-1]+v)
    sequence -= (np.max(sequence) + np.min(sequence))/2
    sequence /= np.max(np.abs(sequence))
    return sequence

def fm_sinwave(freq, phase, fm_amp, fm_freq, sampling_freq):
    time_series = np.linspace(0,1,sampling_freq)
    phase_series = time_series * freq + phase/2/np.pi + fm_amp * np.sin(fm_freq*time_series*np.pi*2)
    sequence = np.sin( phase_series*2*np.pi )
    return sequence

def generate_echo_sequence(sequence, delay):
    sequence = np.roll(sequence,delay)
    if delay > 0:
        sequence[:delay]=0
    elif delay < 0:
        sequence[delay:]=0
    return sequence

class SimpleRecurrentNeuralNetwork(object):
    def __init__(self):
        self.session = None
    
    def __del__(self):
        if self.session is not None:
            self.session.close()
    
    def __feed_forward(self, input_sequence, initial_state):
        W_state = tf.Variable(tf.random_normal([self.hidden_unit_count+2, self.hidden_unit_count], dtype=tf.float64), dtype=tf.float64)
        W_out = tf.Variable(tf.random_normal([self.hidden_unit_count+1, 1], dtype=tf.float64), dtype=tf.float64)
        state = initial_state
        unstacked_input_sequence = tf.unstack(input_sequence, axis=1)
        unstacked_output_sequence = []
        for input_value in unstacked_input_sequence:
            input_value = tf.reshape(input_value, [-1, 1])
            bias_value = np.ones([self.sequence_count,1])
            concatenated_state = tf.concat([state, input_value, bias_value], axis=1)
            state = tf.tanh(tf.matmul(concatenated_state, W_state))
            bias_value = np.ones([self.sequence_count,1])
            concatenated_state = tf.concat([state, bias_value], axis=1)
            output_value = tf.matmul(concatenated_state, W_out)
            unstacked_output_sequence.append(output_value)
        final_state = state
        output_sequence = tf.stack(unstacked_output_sequence, axis=1)
        output_sequence = tf.squeeze(output_sequence,axis=2)
        return output_sequence, final_state

    def generate_batch(self, sequence_list, batch_length):
        sequence_count = sequence_list.shape[0]
        sequence_batch = np.reshape(sequence_list, [sequence_count,-1,batch_length])
        sequence_batch = np.transpose(sequence_batch,[1,0,2])
        return sequence_batch
    
    def train(self, input_sequence_list, output_sequence_list, hidden_unit_count, batch_length=10, learning_rate=1e-5, epoch_count=500):
        assert(input_sequence_list.shape == output_sequence_list.shape)
        if self.session is not None:
            self.session.close()
        tf.reset_default_graph()
        self.sequence_count, self.sequence_length = input_sequence_list.shape
        self.hidden_unit_count = hidden_unit_count
        self.batch_length = batch_length
        self.batch_count = self.sequence_length//self.batch_length
        self.input_sequence_op = tf.placeholder(tf.float64, [None, self.batch_length])
        self.output_sequence_op = tf.placeholder(tf.float64, [None, self.batch_length])
        self.initial_state_op = tf.placeholder(tf.float64, [None, self.hidden_unit_count])
        self.prediction_sequence_op, self.last_state_op = self.__feed_forward(self.input_sequence_op, self.initial_state_op)
        self.loss_op = tf.nn.l2_loss(self.prediction_sequence_op - self.output_sequence_op)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
        self.train_op = optimizer.minimize(self.loss_op/self.sequence_count)
        init_op = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init_op)
        self.loss_history = []
        input_sequence_batch = self.generate_batch(input_sequence_list, self.batch_length)
        output_sequence_batch = self.generate_batch(output_sequence_list, self.batch_length)
        epoch_range = tqdm.trange(epoch_count)
        for _ in epoch_range:
            last_state = np.zeros([self.sequence_count,self.hidden_unit_count])
            loss_sum = 0
            for batch_index in range(self.batch_count):
                feed_dict = {
                    self.input_sequence_op: input_sequence_batch[batch_index],
                    self.output_sequence_op: output_sequence_batch[batch_index],
                    self.initial_state_op: last_state
                }
                _, loss, last_state = self.session.run([self.train_op, self.loss_op, self.last_state_op], feed_dict = feed_dict)
                loss_sum += loss/self.sequence_count
            self.loss_history.append(loss_sum)
            epoch_range.set_description("loss=%.3f"%loss_sum)

    def predict(self, input_sequence_list, output_sequence_list):
        sequence_count, sequence_length = input_sequence_list.shape
        input_sequence_batch = self.generate_batch(input_sequence_list, self.batch_length)
        output_sequence_batch = self.generate_batch(output_sequence_list, self.batch_length)
        batch_count = sequence_length // self.batch_length
        last_state = np.zeros([sequence_count, self.hidden_unit_count])
        prediction_sequence_batch_list = []
        loss_sum = 0
        for batch_index in range(batch_count):
            feed_dict = {
                self.input_sequence_op: input_sequence_batch[batch_index],
                self.output_sequence_op: output_sequence_batch[batch_index],
                self.initial_state_op: last_state
            }
            prediction_sequence_batch, last_state, loss = self.session.run([self.prediction_sequence_op, self.last_state_op, self.loss_op], feed_dict=feed_dict)
            prediction_sequence_batch_list.append(prediction_sequence_batch)
            loss_sum += loss/sequence_count
        prediction_sequence_list = np.hstack(prediction_sequence_batch_list)
        return prediction_sequence_list, loss_sum


def generate_data(sequence_count, sequence_length, delay):
    input_sequence_list = []
    for sequence_index in range(sequence_count):
        r = sequence_index % 6
        if r == 0:
            input_sequence = sinwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r == 1:
            input_sequence = sqwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r == 2:
            input_sequence = triwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r == 3:
            input_sequence = sawwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, sequence_length)
        elif r == 4:
            input_sequence = moving_noise(np.random.rand(), np.random.rand(), np.random.rand(), sequence_length)
        elif r == 5:
            input_sequence = fm_sinwave(3.+np.random.rand()*3, np.random.rand()*2*np.pi, np.random.rand(), np.random.rand()*5, sequence_length)
        else:
            assert(0 <= r and r < 6)
            exit(0)
        input_sequence_list.append(input_sequence)
    output_sequence_list = []
    for input_sequence in input_sequence_list:
        output_sequence = generate_echo_sequence(input_sequence, delay)
        output_sequence_list.append(output_sequence)

    input_sequence_list = np.array(input_sequence_list)
    output_sequence_list = np.array(output_sequence_list)

    assert(input_sequence_list.shape == (sequence_count, sequence_length))
    assert(output_sequence_list.shape == (sequence_count, sequence_length))
    return input_sequence_list, output_sequence_list

class ReservoirComputing(object):
    def __feed_forward(self, input_sequence_list):
        sequence_count, sequence_length = input_sequence_list.shape
        predict_time_series = []
        state = np.zeros([input_sequence_list.shape[0], self.hidden_unit_count])
        state_list = []
        for sequence_index in range(sequence_length):
            input_value_list = input_sequence_list[:,sequence_index]
            input_value_list = np.expand_dims(input_value_list, axis=1)
            stacked_state = np.hstack([state,input_value_list,np.ones([sequence_count,1])])
            state = np.tanh(stacked_state @ self.W_state)
            state_list.append(state)
            stacked_state = np.hstack([state, np.ones([sequence_count,1])])
            predict_value_list = stacked_state @ self.W_out
            predict_value_list = np.squeeze(predict_value_list, axis=1)
            predict_time_series.append(predict_value_list)
        predict_sequence_list = np.transpose(np.array(predict_time_series), [1,0])
        state_list = np.transpose(np.array(state_list),[1,0,2])
        return predict_sequence_list, state_list

    def train(self, input_sequence_list, output_sequence_list, hidden_unit_count, radius, beta):
        assert(input_sequence_list.shape == output_sequence_list.shape)
        self.hidden_unit_count = hidden_unit_count
        self.radius = radius
        self.sequence_count, self.sequence_length = input_sequence_list.shape
        self.hidden_unit_count = hidden_unit_count
        self.W_state = np.random.rand(self.hidden_unit_count+2, self.hidden_unit_count)
        self.W_out = np.random.rand(self.hidden_unit_count+1,1)
        norm = np.max(np.abs(np.linalg.eig(self.W_state[:self.hidden_unit_count,:])[0]))
        self.W_state *= self.radius/norm
        _,state_list = self.__feed_forward(input_sequence_list)
        state_list = np.array(state_list)
        V = np.reshape(state_list, [-1, hidden_unit_count])
        V = np.hstack([V,np.ones([V.shape[0], 1])])
        S = np.reshape(output_sequence_list, [-1])
        self.W_out = np.linalg.pinv(V,rcond = beta) @ S
        self.W_out = np.expand_dims(self.W_out,axis=1)
        
    def predict(self, input_sequence_list, output_sequence_list):
        prediction_sequence_list, _ = self.__feed_forward(input_sequence_list)
        loss = np.sum((prediction_sequence_list-output_sequence_list)**2)/2
        loss /= prediction_sequence_list.shape[0]
        return prediction_sequence_list, loss



def plot(input_sequence_list, output_sequence_list, prediction_sequence_list):
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(12,10))
    for index in range(6):
        plt.subplot(6,1,index+1)
        plt.plot(input_sequence_list[index],color=cmap(index),linestyle="--",label="input")
        plt.plot(output_sequence_list[index],color=cmap(index),linestyle=":",label="label")
        plt.plot(prediction_sequence_list[index],color=cmap(index),label="prediction")
        plt.legend()
    plt.show()


sequence_count = 100
sequence_length = 400
delay = 5
input_sequence_list, output_sequence_list = generate_data(sequence_count, sequence_length, delay)
'''model = SimpleRecurrentNeuralNetwork()
batch_length = 10
epoch_count = 300
learning_rate = 1e-3
hidden_unit_count = 10
model.train(input_sequence_list, output_sequence_list, hidden_unit_count, learning_rate=learning_rate, batch_length=batch_length, epoch_count=epoch_count)
prediction_sequence_list, loss = model.predict(input_sequence_list, output_sequence_list)
print("loss=%f"%loss)
plot(input_sequence_list, output_sequence_list, prediction_sequence_list)'''
hidden_unit_count = 100
radius = 1.25
beta = 1e-14
model = ReservoirComputing()
start_time = time.time()
model.train(input_sequence_list, output_sequence_list, hidden_unit_count, radius, beta)
print("elapsed %f sec"%(time.time()-start_time))
prediction_sequence_list, loss = model.predict(input_sequence_list,output_sequence_list)
print("loss=%f"%loss)
plot(input_sequence_list, output_sequence_list, prediction_sequence_list)
#for checking about overtraining
'''test_input_sequence_list, test_output_sequence_list = generate_data(sequence_count = sequence_count, sequence_length = sequence_length, delay=delay)
test_prediction_sequence_list, test_loss = model.predict(test_input_sequence_list,test_output_sequence_list)
print("test_loss=%f"%test_loss)
plot(test_input_sequence_list, test_output_sequence_list, test_prediction_sequence_list)'''