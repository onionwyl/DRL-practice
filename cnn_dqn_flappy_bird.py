import tensorflow as tf
from tensorflow._api.v1 import keras
from tensorflow._api.v1.keras import layers
from flappy_bird_utils.flappy_bird import FlappyBird
from collections import deque
import numpy as np
import random

class dqn_bot():
    def __init__(self, env):
        self.env = env
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_model()

        # Memory buffer
        self.memory_buffer = deque()
        self.memory_size = 100000
        # discount rate
        self.gamma = 0.95
        # random action rate
        self.epsilon = 0.9
        # random action rate decay
        self.epsilon_decay = 0.9995
        self.observe = 10000
        self.max_score = 0

    def build_model(self):
        inputs = keras.Input(shape=(80, 80, 4, ))
        x = layers.Conv2D(32, (8, 8), (4, 4), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2), (2, 2), padding="same")(x)
        x = layers.Conv2D(64, (4, 4), (2, 2), padding="same", activation="relu")(x)
        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dense(2, activation="linear")(x)
        return keras.Model(inputs=inputs, outputs=x)

    def update_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def move(self, state):
        if np.random.rand() < self.epsilon:
            print("haha")
            return np.random.randint(0, 2)
        else :
            return np.argmax(self.model.predict(state)[0])
    def train(self, episode, batch):
        self.model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
        # count
        count = 0
        rewards = []
        for i in range(episode):
            state, _, _ = env.frame_step(0)
            state = np.reshape(state, (-1, 80, 80))
            state_t = np.stack((state, state, state, state), axis=3)
            reward_sum = 0
            loss = np.infty
            dead = False
            while not dead:
                if env.score > self.max_score:
                    self.max_score = env.score
                # Model choose an action using ε-greedy
                action = self.move(state_t)
                print(action)
                # Get next state, reward from env
                state, reward, dead = env.frame_step(action)
                # Add state to stacked state
                state = np.reshape(state, (-1, 80, 80, 1))
                state_t_next = np.append(state, state_t[:, :, :, :3], axis=3)
                # Add data to memory
                self.memory_buffer.append([state_t[0], action, reward, state_t_next[0], dead])
                reward_sum += reward
                if len(self.memory_buffer) > self.memory_size:
                    self.memory_buffer.popleft()
                if len(self.memory_buffer) > self.observe:
                    # train
                    X, y = self.get_batch(batch)
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                    self.epsilon *= self.epsilon_decay
                    if count != 0 and count % 5 == 0:
                        self.update_model()
                state_t = state_t_next
            rewards.append(reward_sum)
            if i % 10 == 0:
                if len(self.memory_buffer) > self.observe:
                    print("Training")
                    print("Explore Rate: " + str(self.epsilon))
                    print("Max Score: " + str(self.max_score))
                    print("--------------")
                else:
                    print("Observing")


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
    bot = dqn_bot(env)
    bot.train(10000, 32)
    play(env, bot)