import tensorflow as tf
from tensorflow._api.v1 import keras
from tensorflow._api.v1.keras import layers
from flappy_bird_utils.flappy_bird import FlappyBird
from collections import deque
import numpy as np
import random
class DqnBot(object):
    def __init__(self, env):

        self.env = env

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_model()

        # Memory buffer
        self.memory_buffer = deque(maxlen=2000)
        # discount rate
        self.gamma = 0.95
        # random action rate
        self.epsilon = 0.9
        # random action rate decay
        self.epsilon_decay = 0.9995

    def build_model(self):
        inputs = keras.Input(shape=(3,))
        x = layers.Dense(16, activation="relu")(inputs)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(2, activation="linear")(x)
        return keras.Model(inputs = inputs, outputs = x)

    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def load(self):
        pass
    def save(self):
        pass
    def move(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)
        else :
            return np.argmax(self.model.predict(state)[0])

    def train(self, episode, batch):
        self.model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
        # count
        count = 0

        for i in range(episode):
            state = self.make_state(self.env.get_q_state())
            reward_sum = 0
            loss = np.infty
            dead = False
            while not dead:
                x = state.reshape((-1, 3))
                # Model choose an action using ε-greedy
                action = self.move(x)
                # Get reward from env
                _, _, dead = env.frame_step(action)
                if not dead:
                    reward = 1
                else:
                    reward = -5
                # Get next state
                state = self.make_state(self.env.get_q_state())
                # Add data to memory
                self.memory_buffer.append([x[0], action, reward, state, dead])
                reward_sum += reward
                if len(self.memory_buffer) > batch:
                    # train
                    X, y = self.get_batch(batch)
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                    self.epsilon *= self.epsilon_decay
                    if count != 0 and count % 5 == 0:
                        self.update_model()
            # if i % 5 == 0:


    def get_batch(self, batch):
        batch_data = random.sample(self.memory_buffer, batch)
        states = np.array([d[0] for d in batch_data])
        next_states = np.array([d[3] for d in batch_data])
        y = self.model.predict(states)
        q_next = self.target_model.predict(next_states)
        for i, (_, action, reward, _, dead) in enumerate(batch_data):
            y[i][action] = reward + self.gamma * np.amax(q_next[i]) * (1 - dead)
        return states, y




    def make_state(self, state):
        [player_x, player_y, player_vel_y, pipes, pipe_width] = state
        index = 0
        for i in range(len(pipes)):
            if pipes[i]['x'] + pipe_width >= player_x:
                index = i
                break
        dx = int(pipes[index]['x'] - player_x)
        dy = int(pipes[index]['y'] - player_y)
        return np.array([dx, dy, player_vel_y])

def play(env, bot):
    pass


if __name__ == "__main__":
    env = FlappyBird()
    bot = DqnBot(env)
    bot.train(10000, 128)
    play(env, bot)

