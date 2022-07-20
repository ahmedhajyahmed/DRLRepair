from random import random

from Clean_code.DQN import DeepQNetwork
import gym
import matplotlib.pyplot as plt
import numpy as np

from Clean_code.RL_runner import RL_runner


def visualize(env, rewards, epsilons, EPISODES, TRAIN_END):
    rolling_average = np.convolve(rewards, np.ones(100) / 100)
    plt.plot(rewards)
    plt.plot(rolling_average, color='black')
    plt.axhline(y=195, color='r', linestyle='-')  # Solved Line
    # Scale Epsilon (0.001 - 1.0) to match reward (0 - 200) range
    eps_graph = [200 * x for x in epsilons]
    plt.plot(eps_graph, color='g', linestyle='-')
    # Plot the line where TESTING begins
    plt.axvline(x=TRAIN_END, color='y', linestyle='-')
    plt.xlim((0, EPISODES))
    plt.ylim((0, 220))
    plt.show()
    env.close()


def create_agent(env, batch_size):
    # Create the agent
    nS = env.observation_space.shape[0]  # This is only 4
    nA = env.action_space.n  # Actions
    return DeepQNetwork(batch_size, nS, nA)


def save_model(dqn, path, internals_path=None):
    dqn.save(path)
    if internals_path is not None:
        dqn.save_internals(internals_path)


def random_drift_cartpole_env(force_mag_ranges=(0.0, 30.0), mass_cart_ranges=(1.0, 50.0), mass_pole_ranges=(0.1, 50.0),
                              length_ranges=(0.5, 10.0)):
    env_changed = gym.make('CartPole-v1').unwrapped
    env_changed.force_mag = random.uniform(force_mag_ranges[0], force_mag_ranges[1])  # default is +10.0
    env_changed.masscart = random.uniform(mass_cart_ranges[0], mass_cart_ranges[1])  # default is 1
    env_changed.masspole = random.uniform(mass_pole_ranges[0], mass_pole_ranges[1])  # default is 0.1
    env_changed.length = random.uniform(length_ranges[0], length_ranges[1])  # default is 0.5
    return env_changed


def drift_cartpole_env(force_mag=10, mass_cart=1, mass_pole=0.1, length=0.5):
    env_changed = gym.make('CartPole-v1').unwrapped
    env_changed.force_mag = force_mag  # default is +10.0
    env_changed.masscart = mass_cart  # default is 1
    env_changed.masspole = mass_pole  # default is 0.1
    env_changed.length = length  # default is 0.5
    return env_changed


# def main(batch_size=24):
#     envCartPole = gym.make('CartPole-v1')
#     agent = create_agent(envCartPole, batch_size)
#     rl_runner = RL_runner(agent=agent, env=envCartPole)
#
#     print("------------------training---------------------")
#     test_episodes = rl_runner.train(envCartPole, batch_size)
#     final_rewards, final_epsilons = rl_runner.test(test_episodes=test_episodes, rewards=rl_runner.rewards,
#                                                    epsilons=rl_runner.epsilons)
#     visualize(envCartPole, final_rewards, final_epsilons, rl_runner.EPISODES, rl_runner.TRAIN_END)
#     save_model(dqn=rl_runner.dqn, path="dqn.h5", internals_path="internals.npz")
#
#     print("------------------environment distortion---------------------")
#     rl_runner.dqn.load_dqn("dqn.h5")
#     rewards, epsilons, set_on_training = rl_runner.test(test_episodes=2, rewards=rl_runner.rewards,
#                                                         epsilons=rl_runner.epsilons)
#     env = drift_env()
#     rl_runner.env = env
#     new_rewards, new_epsilons, set_after_shift = rl_runner.test(test_episodes=2, rewards=rl_runner.rewards,
#                                                                 epsilones=rl_runner.epsilons)
#     # visualize(env_changed, new_rewards, new_epsilons)
#
#     print("------------------retraining---------------------")
#     nS = rl_runner.retrain()
#     retraining_rewards, retraining_epsilons, set_retrained_tested_on_changed_env = rl_runner.test(test_episodes=2,
#                                                                                                   rewards=rl_runner.rewards,
#                                                                                                   epsilons=rl_runner.epsilons)
#     # visualize(env_changed, retraining_rewards, retraining_epsilons)
#     retraining_rewards, retraining_epsilons, set_retrained_tested_on_org_env = rl_runner.test(test_episodes=2,
#                                                                                               rewards=rl_runner.rewards,
#                                                                                               epsilons=rl_runner.epsilons)
#     # visualize(env_changed, retraining_rewards, retraining_epsilons)
#     save_model(dqn=rl_runner.dqn, path="./experiments/re_dqn_fm_500_m_500.h5")
#     # new_dqn.save_internals("./experiments/internals_fm_5_m_5.npz")
#     print("set_on_training vs set_after_shift :",
#           len(set_on_training.intersection(set_after_shift)) / len(set_on_training.union(set_after_shift)))
#     print("set_on_training vs set_after _shift :",
#           len(set_on_training.intersection(set_retrained_tested_on_changed_env)) /
#           len(set_on_training.union(set_retrained_tested_on_changed_env)))

def main(batch_size=24):
    env = drift_cartpole_env()

    for i in range(5):
        envCartPole = gym.make('CartPole-v1')
        agent = create_agent(envCartPole, batch_size)
        rl_trainer = RL_runner(agent=agent, env=envCartPole, EPISODES=5)

        print("------------------training---------------------")
        test_episodes = rl_trainer.train(envCartPole, batch_size)
        final_rewards, final_epsilons = rl_trainer.test(test_episodes=test_episodes, rewards=rl_trainer.rewards,
                                                        epsilons=rl_trainer.epsilons)
        visualize(envCartPole, final_rewards, final_epsilons, rl_trainer.EPISODES, rl_trainer.TRAIN_END)
        save_model(dqn=rl_trainer.dqn, path="dqn_run_" + str(i) + ".h5")

        for j in range(5):
            print("------------------Retraining---------------------")
            envCartPole = gym.make('CartPole-v1')
            agent = create_agent(envCartPole, batch_size)
            rl_retrainer = RL_runner(agent=agent, env=envCartPole, EPISODES=3)
            rl_retrainer.dqn.load_dqn("dqn_run_" + str(i) + ".h5")

            # final_rewards, final_epsilons = rl_trainer.test(test_episodes=2, rewards=rl_runner.rewards,
            #                                                epsilons=rl_runner.epsilons)
            # visualize(envCartPole, final_rewards, final_epsilons, rl_runner.EPISODES, rl_runner.TRAIN_END)

            env = drift_cartpole_env()
            rl_retrainer.env = env

            # final_rewards, final_epsilons = rl_runner.test(test_episodes=2, rewards=rl_runner.rewards,
            #                                                epsilons=rl_runner.epsilons)
            # visualize(env, final_rewards, final_epsilons, rl_runner.EPISODES, rl_runner.TRAIN_END)

            rl_retrainer.retrain(batch_size)

            # final_rewards, final_epsilons = rl_runner.test(test_episodes=2, rewards=rl_runner.rewards,
            #                                                epsilons=rl_runner.epsilons)
            # visualize(env, final_rewards, final_epsilons, rl_runner.EPISODES, rl_runner.TRAIN_END)

            final_rewards, final_epsilons = rl_retrainer.test(test_episodes=2, rewards=rl_retrainer.rewards,
                                                              epsilons=rl_retrainer.epsilons)
            # visualize(env_changed, retraining_rewards, retraining_epsilons)
            save_model(dqn=rl_retrainer.dqn, path="./experiments/re_dqn_" + str(i) + "_" + str(j) + ".h5")
            # new_dqn.save_internals("./experiments/internals_fm_5_m_5.npz")


if __name__ == '__main__':
    main()
