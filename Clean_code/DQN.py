import numpy as np
from collections import deque
from tensorflow import keras
import random


class DeepQNetwork:
    def __init__(self, batch_size, states, actions,   alpha =  0.001, gamma = 0.95, epsilon = 1, epsilon_min = 0.001, epsilon_decay =  0.995):
        self.nS = states
        self.nA = actions
        self.memory = deque([], maxlen=2500)
        self.alpha = alpha
        self.gamma = gamma
        # Explore/Exploit
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self.build_model()
        self.loss = []
        self.batch_size = batch_size
        self.WEIGHTS = []
        self.BIASES = []
        self.OBSERVATIONS = []

    def build_model(self):
        model = keras.Sequential()  # linear stack of layers https://keras.io/models/sequential/
        model.add(keras.layers.Dense(24, input_dim=self.nS, activation='relu'))  # [Input] -> Layer 1
        #   Dense: Densely connected layer https://keras.io/layers/core/
        #   24: Number of neurons
        #   input_dim: Number of input variables
        #   activation: Rectified Linear Unit (relu) ranges >= 0
        model.add(keras.layers.Dense(24, activation='relu'))  # Layer 2 -> 3
        model.add(keras.layers.Dense(self.nA, activation='linear'))  # Layer 3 -> [output]
        #   Size has to match the output (different actions)
        #   Linear activation on the last layer
        model.compile(loss='mean_squared_error',  # Loss function: Mean Squared Error
                      optimizer=keras.optimizers.Adam(
                          lr=self.alpha))  # Optimizer: Adam (Feel free to check other options)
        return model

    def action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)  # Explore
        action_vals = self.model.predict(state)  # Exploit: Use the NN to predict the correct action from this state
        return np.argmax(action_vals[0])

    def test_action(self, state):  # Exploit
        action_vals = self.model.predict(state)
        return np.argmax(action_vals[0])

    def store(self, state, action, reward, nstate, done):
        # Store the experience in memory
        self.memory.append((state, action, reward, nstate, done))

    def save(self, file_name):
        self.model.save(file_name)

    def load_dqn(self, file_path):
        self.model = keras.models.load_model(file_path)

    def experience_replay(self):
        # Execute the experience replay
        minibatch = random.sample(self.memory, self.batch_size)  # Randomly sample from memory

        # Convert to numpy for speed by vectorization
        x = []
        y = []
        np_array = np.array(minibatch)

        # track
        self.OBSERVATIONS.append(np_array)

        st = np.zeros((0, self.nS))  # States
        nst = np.zeros((0, self.nS))  # Next States
        for i in range(len(np_array)):  # Creating the state and next state np arrays
            st = np.append(st, np_array[i, 0], axis=0)
            nst = np.append(nst, np_array[i, 3], axis=0)
        st_predict = self.model.predict(st)  # Here is the speedup! I can predict on the ENTIRE batch
        nst_predict = self.model.predict(nst)
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            # Predict from state
            nst_action_predict_model = nst_predict[index]
            if done == True:  # Terminal: Just assign reward much like {* (not done) - QB[state][action]}
                target = reward
            else:  # Non-terminal
                target = reward + self.gamma * np.amax(nst_action_predict_model)
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1
        # Reshape for Keras Fit
        x_reshape = np.array(x).reshape(self.batch_size, self.nS)
        y_reshape = np.array(y)
        epoch_count = 1  # Epochs is the number or iterations
        hist = self.model.fit(x_reshape, y_reshape, epochs=epoch_count, verbose=0)
        # track weights and biases
        self.track_internals()
        # Graph Losses
        for i in range(epoch_count):
            self.loss.append(hist.history['loss'][i])
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def track_internals(self):
        tamp = []
        for layer in self.model.layers:
            tamp.append(np.array(layer.weights[0]))
            tamp.append(np.array(layer.weights[1]))
        self.WEIGHTS.append(tamp)

    def save_internals(self, file_name):
        a = self.model.layers[0].weights
        observations = np.array(self.OBSERVATIONS)
        nb_layers = len(self.model.weights)
        weights = []
        for n in range(nb_layers):
            weights.append([])
        for step in self.WEIGHTS:
            for n in range(nb_layers):
                weights[n].append(np.array(step[n]))
        internals_dict = {'ob': observations}
        for n in range(nb_layers):
            internals_dict[str(n)] = np.array(weights[n])

        np.savez(file_name, **internals_dict)
