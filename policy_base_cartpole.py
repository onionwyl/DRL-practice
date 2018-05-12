import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1")

s = env.reset()
s_size = s.shape[0]
a_size = 2
# discount rate
gamma = 0.99

# agent
input = tf.placeholder(tf.float32, [None, s_size])
W1 = tf.Variable(tf.truncated_normal([s_size, 8]), dtype=tf.float32, name="W1")
b1 = tf.Variable(tf.zeros([8]), name="b1")
h1 = tf.nn.relu(tf.matmul(input, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([8, a_size], dtype=tf.float32, name="W2"))
b2 = tf.Variable(tf.zeros([a_size], name="b2"))
outputs = tf.nn.softmax(tf.matmul(h1, W2) + b2)

reward_holder = tf.placeholder(tf.float32, [None])
action_holder = tf.placeholder(tf.int32, [None])
# select the responsible outputs from all outputs
# index of responsible outputs
indexes = tf.range(0, tf.shape(outputs)[0]) * tf.shape(outputs[1]) + action_holder
# gather responsible outputs
responsible_outputs = tf.gather(tf.reshape(outputs, [-1]), indexes)
loss = -tf.reduce_mean(tf.log(responsible_outputs) * reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
# Accumulate gradients
tvars = tf.trainable_variables()
gradient_holders = [tf.placeholder(tf.float32, name=str(idx)+'_holder') for idx, var in enumerate(tvars)]
gradients = tf.gradients(loss, tvars)
update_batch = optimizer.apply_gradients(zip(gradient_holders, tvars))

# training process
num_episodes = 5000
max_exp = 1000
update_fq = 5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Use MC method
    total_reward = []
    total_step = []
    grad_buffer = sess.run(tf.trainable_variables())
    for idx, grad in enumerate(grad_buffer):
        grad_buffer[idx] = grad * 0
    for i in range(num_episodes):
        s = env.reset()
        tmp_reward = 0
        exp = []
        for j in range(max_exp):
            action_vector = sess.run(outputs, feed_dict={input: [s]})
            # Randomly choose a action depend on policy(output probability)
            action = np.random.choice(action_vector[0], p=action_vector[0])
            action = np.argmax(action_vector == action)

            s_next, reward, d, _ = env.step(action)
            # env.render()
            exp.append([s, action, reward, s_next])
            s = s_next
            tmp_reward += reward
            # end of a episode
            if d == True:
                # update network
                exp = np.array(exp)
                # accumulate discounted reward
                r_vector = exp[:, 2]
                discounted_r = np.zeros_like(r_vector)
                tmp = 0
                for t in reversed(range(r_vector.size)):
                    tmp = tmp * gamma + r_vector[t]
                    discounted_r[t] = tmp
                exp[:, 2] = discounted_r
                grads = sess.run(gradients, feed_dict={reward_holder:exp[:, 2], input:np.vstack(exp[:, 0]), action_holder:exp[:, 1]})
                for idx, grad in enumerate(grads):
                    grad_buffer[idx] += grad
                if i % update_fq == 0 and i != 0:
                    _ = sess.run(update_batch, feed_dict=dict(zip(gradient_holders, grad_buffer)))
                    for idx, grad in enumerate(grad_buffer):
                        grad_buffer[idx] = grad * 0
                total_reward.append(tmp_reward)
                total_step.append(j)
                break
        if i % 100 == 0:
            print(np.mean(total_reward[-100:]))
            if np.mean(total_reward[-100:]) > 300:
                break
    s = env.reset()
    env.render()
    while(True):
        action_vector = sess.run(outputs, feed_dict={input: [s]})
        action = np.argmax(action_vector)
        s_next, reward, d, _ = env.step(action)
        env.render()
        s = s_next
        if d == True:
            s = env.reset()
            env.render()





