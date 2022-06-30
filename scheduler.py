# Imports
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
import random
import os
from tknc.tknc_coverage import Coverage

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Create Gym
from gym import wrappers

envCartPole = gym.make('CartPole-v1')
envCartPole.seed(50)  # Set the seed to keep the environment consistent across runs

# Global Variables
# EPISODES = 500
EPISODES = 150
TRAIN_END = 0

# Hyper Parameters
def discount_rate():  # Gamma
    return 0.95


def learning_rate():  # Alpha
    return 0.001


def get_batch_size():  # Size of the batch used in the experience replay
    return 24


class DeepQNetwork:
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay, batch_size):
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
        self.batch_size = batch_size
        self.WEIGHTS = []
        self.BIASES = []
        self.OBSERVATIONS = []
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha)

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
        return model

    def compute_loss(self, x, y, weights, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self.model(x, training=training)
        a = self.loss_object(y_true=y, y_pred=y_, sample_weight=weights)
        b = self.loss_object(y_true=y, y_pred=y_)
        return self.loss_object(y_true=y, y_pred=y_, sample_weight=weights)

    def compute_grad(self, inputs, targets, weights):
        with tf.GradientTape() as tape:
            loss_value = self.compute_loss(inputs, targets, weights, training=True)
        return loss_value, tape.gradient(loss_value, self.model.trainable_variables)

    def customized_fit(self, x, y, weights):
        # Optimize the model
        loss_value, grads = self.compute_grad(x, y, weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value, grads

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

        hist_loss, hist_grad = self.customized_fit(x_reshape, y_reshape, np.ones(shape=x_reshape.shape[0])* 0.5)
        # print("epoch: {}; loss: {};".format(epoch_count, hist_loss))

        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return hist_loss


def train(env, batch_size):
    train_loss_results = []
    epoch_loss_avg = None
    # Create the agent
    nS = env.observation_space.shape[0]  # This is only 4
    nA = env.action_space.n  # Actions
    dqn = DeepQNetwork(nS, nA, learning_rate(), discount_rate(), 1, 0.001, 0.995, batch_size)

    # Training
    rewards = []  # Store rewards for graphing
    epsilons = []  # Store the Explore/Exploit
    test_episodes = 0
    for e in range(EPISODES):
        epoch_loss_avg = tf.keras.metrics.Mean()
        state = env.reset()
        state = np.reshape(state, [1, nS])  # Resize to store in memory to pass to .predict
        tot_rewards = 0
        for time in range(210):  # 200 is when you "solve" the game. This can continue forever as far as I know
            action = dqn.action(state)
            nstate, reward, done, _ = env.step(action)
            nstate = np.reshape(nstate, [1, nS])
            tot_rewards += reward
            dqn.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
            state = nstate
            # done: CartPole fell.
            # time == 209: CartPole stayed upright
            if done or time == 209:
                rewards.append(tot_rewards)
                epsilons.append(dqn.epsilon)
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e, EPISODES, tot_rewards, dqn.epsilon))
                break
            # Experience Replay
            if len(dqn.memory) > batch_size:
                loss_value = dqn.experience_replay()
                epoch_loss_avg.update_state(loss_value)
        # If our current NN passes we are done
        # I am going to use the last 5 runs
        if len(rewards) > 5 and np.average(rewards[-5:]) > 195:
            # Set the rest of the EPISODES for testing
            test_episodes = EPISODES - e
            TRAIN_END = e
            break
        train_loss_results.append(epoch_loss_avg.result())

    return test_episodes, nS, dqn, rewards, epsilons, train_loss_results


def test(env, test_episodes, nS, dqn, rewards, epsilons):
    # Test the agent that was trained
    #   In this section we ALWAYS use exploit don't train anymore
    test_set = []
    for e_test in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, nS])
        tot_rewards = 0
        for t_test in range(210):
            action = dqn.test_action(state)
            test_set.append(state)
            n_state, reward, done, _ = env.step(action)
            n_state = np.reshape(n_state, [1, nS])
            tot_rewards += reward
            # DON'T STORE ANYTHING DURING TESTING
            state = n_state
            # done: CartPole fell.
            # t_test == 209: CartPole stayed upright
            if done or t_test == 209:
                rewards.append(tot_rewards)
                epsilons.append(0)  # We are doing full exploit
                print("episode: {}/{}, score: {}, e: {}"
                      .format(e_test, test_episodes, tot_rewards, 0))
                break

    return rewards, epsilons


def visualize(env, rewards, train_loss_results):
    rolling_average = np.convolve(rewards, np.ones(100) / 100)

    plt.plot(rewards)
    plt.plot(rolling_average, color='black')
    plt.axhline(y=195, color='r', linestyle='-')  # Solved Line
    # Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range

    plt.plot(train_loss_results, color='g', linestyle='-')
    plt.xlim((0, EPISODES))
    plt.ylim((0, 220))
    plt.show()

    env.close()


def main():
    print("------------------training---------------------")
    batch_size = get_batch_size()
    test_episodes, nS, dqn, rewards, epsilons, losses = train(envCartPole, batch_size)
    final_rewards, final_epsilons = test(envCartPole, 10, nS, dqn, rewards, epsilons)
    visualize(envCartPole, final_rewards, losses)


if __name__ == '__main__':
    main()
