import numpy as np
import tensorflow as tf
import gym

num_episodes = 5000
max_exp = 1000
gamma = 0.99
learning_rate_actor = 0.001
learning_rate_critic = 0.01

env = gym.make('CartPole-v1')

class Actor(object):
    def __init__(self):
        pass
    def learn(self):
        pass
    def choose_action(self):
        pass

class Critic(object):
    def __init__(self):
        pass
    def learn(self):
        pass

def main():
    pass

if __name__ == "__main__":
    main()
