from keras.datasets import mnist
import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D, Activation
from keras.optimizers import Adam
from copy import deepcopy

(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.set_printoptions(precision=2)

class Environment:
    def __init__(self, X, Y):
        self.X = X # Input dataset
        self.Y = Y # Labels for the images
        self.x = None # image with tranformations for current sessions
        self.y = None # correct class label for image in currect session
        self.max_depth = 12 # The max number of transformations that can be done
        self.memory = [] # Memory for storing images after each transformation
        self.n_stop_actions = 10
        self.n_actions = self.n_stop_actions + 2 + 2 # number of stop actions , 2 rotate actions and 2 flip actions

    def step(self, action):
        if action >= self.n_actions or action < 0: # Invalid Action
            return
        done = False
        reward = 0
        # If it is a stop action or if the length is exceeded, set done
        # as true and reset memory , else append the old memory
        if len(self.memory) + 1 == self.max_depth:
            done = True
            reward= -0.1
        if action < self.n_stop_actions:
            done = True
            if action == self.y: # if the prediction is correct then set reward as n_stop_actions - 1
                reward = self.n_stop_actions - 1
            else:
                reward = -1
            return self.x, reward, done, {}
        if not done: # Else add the image to the memory
            self.memory.append(self.x)
        if action == 10: # If the action is 10, rotate the image by 90 degrees anti-clockwise
            self.x = np.rot90(self.x, 1)
        elif action == 11: # If the action is 11, rotate the image by 90 degrees clockwise
            self.x = np.rot90(self.x, 3)
        elif action == 12: # If the action is 12, horizontally flip the image
            self.x = np.flip(self.x, 1)
        elif action == 13: # If the action is 13, vertically flip the image
            self.x = np.flip(self.x, 0)
        return self.x, reward, done, {}

    def reset(self):
        index = np.random.randint(len(self.X)) # get a random image for the session
        self.x = deepcopy(self.X[index]) # set x as the image from index
        self.y = deepcopy(self.Y[index]) # set the label for image in y
        #clear memory
        del self.memory 
        self.memory = []
        return self.x #return initial image

class DQNCartPoleSolver():
    def __init__(self, n_episodes=1500, gamma=1.0, epsilon=0.99, epsilon_min=0.01, epsilon_log_decay=0.996, batch_size=64):
        self.memory = deque(maxlen=100000)
        self.env = Environment(x_train, y_train)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.n_inputs = 4
        self.n_outputs = 10
        # Init model
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=8,
                              activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, kernel_size=4, activation='relu'))
        self.model.add(Conv2D(64, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(14))
        self.model.compile(loss='mse', optimizer=Adam(lr=1e-6))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        if (np.random.random() <= epsilon):
            return (np.random.randint(self.env.n_actions), self.model.predict(state))
        else:
            return (np.argmax(self.model.predict(state)), self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, self.epsilon)

    def preprocess_state(self, state):
        return np.reshape(state, [1, 28, 28, 1])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + \
                self.gamma * np.max(self.model.predict(next_state))
            x_batch.append(state[0])
            y_batch.append(y_target[0])
        self.model.fit(np.array(x_batch), np.array(y_batch),
                       batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def fit(self):
        totalCorrect = 0
        correct = 0
        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            while not done:
                # self.env.render()
                action, confidence = self.choose_action(
                    state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action, confidence)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done and reward == 9:
                    correct += 1
            if e%20 == 0 and e!=0:
              totalCorrect += correct
              print(
                  '[Episode {}] - correct: {}, accuracy: {}, epsilon: {}.'.format(e, correct, totalCorrect/e, self.epsilon))
              correct=0
            self.replay(self.batch_size)
        return e

    def test(self, n_episodes):
        for episode_num in range(n_episodes):
            done = False
            state = self.preprocess_state(self.env.reset())
            step = 0
            while not done:
                # self.env.render()
                action = self.choose_action(state, 0)
                next_state, _, done, _ = self.env.step(action, 1.0)
                state = self.preprocess_state(next_state)
                step += 1
            print("number of steps in episode {} is {}".format(episode_num, step))

    def save_model(self, filename):
        self.model.save_weights(filename)

    def load_weights(self, filename):
        self.model.load_weights(filename)

from google.colab import drive
agent = DQNCartPoleSolver(n_episodes=1000)  
drive.mount('/content/drive')
agent.model.load_weights("./mnist_reinforcement.h5f")

def predict(x):
	np.argmax(agent.model.predict(x.reshape(1,28,28,1)))

import matplotlib.pyplot as plt
selected = 5073
selected_x = x_test[selected]
selected_y_dash = np.argmax(agent.model.predict(x_test[selected].reshape(1,-1,-1,1)))
selected_y_dash
plt.imshow(x_test[selected])
plt.show()
if selected_y_dash == 10:
  selected_x = np.rot90(selected_x, 1)
elif selected_y_dash == 11:
  selected_x = np.rot90(selected_x, 3)
elif selected_y_dash == 12:
  selected_x = np.flip(selected_x, 1)
elif selected_y_dash == 13:
  selected_x = np.flip(selected_x, 0)
plt.imshow(selected_x)
plt.show()
selected_y_dash = np.argmax(agent.model.predict(selected_x.reshape(1,28,28,1)))
print(selected_y_dash)
