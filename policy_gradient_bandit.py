import tensorflow as tf
import numpy as np

# Define bandits
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pullBandit(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1


# Agent

# network
weight = tf.Variable(tf.ones([num_bandits]))
output = tf.nn.softmax(weight)

# train process
reward_holder = tf.placeholder(tf.float32, [1])
action_holder = tf.placeholder(tf.int32, [1])
loss = -tf.log(tf.slice(output, action_holder, [1])) * reward_holder
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

# train agent
num_episodes = 1000
total_reward = np.zeros(num_bandits)
epsilon = 0.1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_episodes):
        if np.random.randn(1) < epsilon:
            action = np.random.randint(num_bandits)
        else:
            action = np.argmax(sess.run(output))
        reward = pullBandit(bandits[action])
        _, w = sess.run([trainer, weight], feed_dict={reward_holder: [reward], action_holder: [action]})
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
    print("The agent thinks bandit " + str(np.argmax(w)+1) + " is the most promising....")
    if np.argmax(w) == np.argmax(-np.array(bandits)):
        print("...and it was right!")
    else:
        print("...and it was wrong!")

