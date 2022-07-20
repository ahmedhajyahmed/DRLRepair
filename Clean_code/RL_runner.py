import numpy as np


class RL_runner:
    def __init__(self, agent, env, seed=50, EPISODES=150, TRAIN_END=0):
        self.env = env
        self.env.seed(seed)  # Set the seed to keep the environment consistent across runs
        # EPISODES = 500
        self.EPISODES = EPISODES
        self.TRAIN_END = TRAIN_END
        self.dqn = agent
        # Numner of states and actions
        self.nS = self.env.observation_space.shape[0]  # states (only 4 in Cartpole)
        self.nA = self.env.action_space.n  # Actions
        self.rewards = []  # Store rewards for graphing
        self.epsilons = []  # Store the Explore/Exploit

    def train(self, env, batch_size):
        # Training
        test_episodes = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.nS])  # Resize to store in memory to pass to .predict
            tot_rewards = 0
            for time in range(210):  # 200 is when you "solve" the game. This can continue forever as far as I know
                action = self.dqn.action(state)
                nstate, reward, done, _ = env.step(action)
                nstate = np.reshape(nstate, [1, self.nS])
                tot_rewards += reward
                self.dqn.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
                state = nstate
                # done: CartPole fell.
                # time == 209: CartPole stayed upright
                if done or time == 209:
                    self.rewards.append(tot_rewards)
                    self.epsilons.append(self.dqn.epsilon)
                    print("episode: {}/{}, score: {}, e: {}"
                          .format(e, self.EPISODES, tot_rewards, self.dqn.epsilon))
                    break
                # Experience Replay
                if len(self.dqn.memory) > batch_size:
                    self.dqn.experience_replay()
            # If our current NN passes we are done
            # I am going to use the last 5 runs
            if len(self.rewards) > 5 and np.average(self.rewards[-5:]) > 195:
                # Set the rest of the EPISODES for testing
                test_episodes = self.EPISODES - e
                TRAIN_END = e
                break

        return test_episodes

    def test(self, test_episodes, rewards, epsilons):
        # Test the agent that was trained
        #   In this section we ALWAYS use exploit don't train anymore
        test_set = []
        for e_test in range(test_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.nS])
            tot_rewards = 0
            for t_test in range(210):
                action = self.dqn.test_action(state)
                test_set.append(state)
                n_state, reward, done, _ = self.env.step(action)
                n_state = np.reshape(n_state, [1, self.nS])
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

            x_test = np.array(test_set).squeeze()
            # coverage = Coverage(self.dqn.model, None, None)
            # coverage.calculate_metrics(x_test)
            # pattern_set = coverage.pattern_set
        # return rewards, epsilons, pattern_set
        return rewards, epsilons

    def retrain(self, batch_size, retraining_episodes=150):
        nS = self.env.observation_space.shape[0]  # This is only 4

        for e in range(retraining_episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, nS])  # Resize to store in memory to pass to .predict
            tot_rewards = 0
            for time in range(210):  # 200 is when you "solve" the game. This can continue forever as far as I know
                action = self.dqn.action(state)
                nstate, reward, done, _ = self.env.step(action)
                nstate = np.reshape(nstate, [1, nS])
                tot_rewards += reward
                self.dqn.store(state, action, reward, nstate, done)  # Resize to store in memory to pass to .predict
                state = nstate
                # done: CartPole fell.
                # time == 209: CartPole stayed upright
                if done or time == 209:
                    self.rewards.append(tot_rewards)
                    self.epsilons.append(self.dqn.epsilon)
                    print("episode: {}/{}, score: {}, e: {}"
                          .format(e, self.EPISODES, tot_rewards, self.dqn.epsilon))
                    break
                # Experience Replay
                if len(self.dqn.memory) > batch_size:
                    self.dqn.experience_replay()
            # If our current NN passes we are done
            # I am going to use the last 5 runs
            if np.average(self.rewards[-5:]) > 195:
                break

