import numpy as np
import tensorflow as tf
import gym

num_episodes = 5000
max_exp = 1000
gamma = 0.99
learning_rate_actor = 0.001
learning_rate_critic = 0.01
display_reward_threshold = 200
render = False

env = gym.make('CartPole-v1')
num_input = env.observation_space.shape[0]
num_action = env.action_space.n

class Network(object):
    def __init__(self, num_input):
        self.state = tf.placeholder(tf.float32, [1, num_input], "state")
        with tf.variable_scope("Network"):
            self.out = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                name="l1"
            )

class Actor(object):
    def __init__(self, sess, net, num_action, lr=0.001):
        self.sess = sess

        # build Actor Network
        self.action = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        self.net = net
        with tf.variable_scope("Actor"):
            self.outputs = tf.layers.dense(
                inputs=self.net.out,
                units=num_action,
                activation=tf.nn.softmax,
                kernel_initializer=tf.truncated_normal_initializer(),
                bias_initializer=tf.zeros_initializer(),
                name="outputs"
            )
        with tf.variable_scope("exp_value"):
            log_prob = tf.log(self.outputs[0, self.action])
            self.expected_value = tf.reduce_mean(log_prob * self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.expected_value)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.net.state: s, self.action: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.expected_value], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        action_vector = self.sess.run(self.outputs, {self.net.state: s})
        action = np.random.choice(action_vector[0], p=action_vector[0])
        action = np.argmax(action_vector == action)
        return action

class Critic(object):
    def __init__(self, sess, net, lr=0.01):
        self.sess = sess

        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, "reward")
        self.net = net
        with tf.variable_scope('Critic'):
            self.v = tf.layers.dense(
                inputs=self.net.out,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.truncated_normal_initializer(),  # weights
                bias_initializer=tf.zeros_initializer(),  # biases
                name='V'
            )
            with tf.variable_scope('squared_TD_error'):
                self.td_error = self.r + gamma * self.v_ - self.v
                self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
            with tf.variable_scope('train'):
                self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis]
        v_ = self.sess.run(self.v, {self.net.state: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.net.state: s, self.r: r, self.v_: v_})
        return td_error

def main():
    global render
    sess = tf.Session()
    net = Network(num_input)
    actor = Actor(sess, net, num_action)
    critic = Critic(sess, net)
    sess.run(tf.global_variables_initializer())
    total_reward = []
    for i in range(num_episodes):
        s = env.reset()
        step = 0
        tmp_reward = 0
        while True:
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
            if render:
                env.render()
            if done:
                r = -20
            tmp_reward += r
            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)
            s = s_
            step += 1
            if done:
                print("episode:", i, "  reward:", tmp_reward, " step:", step)
                total_reward.append(tmp_reward)
                break
        if i % 100 == 0:
            print(i)
            print(np.mean(total_reward[-100:]))
            if np.mean(total_reward[-100:]) > 300:
                render = True



if __name__ == "__main__":
    main()
