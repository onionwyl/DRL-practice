import tensorflow as tf
from tensorflow._api.v1 import keras
from tensorflow._api.v1.keras import layers
from flappy_bird_utils.flappy_bird import FlappyBird
import numpy as np
import random

class pg_bot():
    def __init__(self, env):
        self.env = env
        self.gamma = 0.8
        self.epsilon = 0
        self.model = self.build_model()
        self.memory_buffer = []
        self.max_score = 0

    def build_model(self):
        inputs = keras.Input(shape=(3,))
        x = layers.Dense(16, activation="relu")(inputs)
        x = layers.Dense(8, activation="relu")(x)
        x = layers.Dense(2, activation="softmax")(x)
        return keras.Model(inputs=inputs, outputs=x)

    def loss(self, y_true, y_pred):
        action_pred = y_pred
        action_true, discount_rewards = y_true[:, 0], y_true[:, 1]
        log_prob = tf.reduce_sum(-tf.log(action_pred) * tf.one_hot(indices=tf.cast(action_true, tf.int32), depth=2), axis=1)
        loss = tf.reduce_mean(log_prob * discount_rewards)
        return loss

    def move(self, state, train = False):
        if train:
            action_prob = self.model.predict(state)[0]
            return np.random.choice(np.array(range(2)), 1, p=action_prob)[0]
        else :
            return np.argmax(self.model.predict(state)[0])

    def discount_reward(self, rewards):
        discount_rewards = np.zeros_like(rewards, dtype=np.float32)
        # 累计的discount reward
        accumulation = 0
        for i in reversed(range(len(rewards))):
            accumulation = accumulation * self.gamma + rewards[i]
            discount_rewards[i] = accumulation

        # normalization 暂时不写
        return list(discount_rewards)


    def train(self, episode, batch):
        self.model.compile(optimizer=keras.optimizers.Adam(), loss=self.loss)

        for i in range(episode):
            state = self.make_state(self.env.get_q_state())
            count = 0
            while True:
                if env.score > self.max_score:
                    self.max_score = env.score
                x = state.reshape((-1, 3))
                action = self.move(x, True)
                _, reward, dead = env.frame_step(action)
                count += 1
                # if not dead and count % 3 == 0:
                #     continue

                # if not dead:
                #     reward = 1
                # else:
                #     reward = -10
                # Get next state
                state = self.make_state(self.env.get_q_state())
                self.memory_buffer.append([x[0], action, reward, state, dead])
                if dead:
                    discount_rewards = self.discount_reward([d[2] for d in self.memory_buffer])
                    break
            if i != 0 and i % batch == 0:
                X = np.array([d[0] for d in self.memory_buffer])
                actions = np.array([d[1] for d in self.memory_buffer])
                y = np.array(list(zip(actions, discount_rewards)))
                loss = self.model.train_on_batch(X, y)
                self.memory_buffer = []
            if i % 100 == 0:
                print("Max score: ", self.max_score)


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
    bot = pg_bot(env)
    bot.train(23333, 2)
    play(env, bot)