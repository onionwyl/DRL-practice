from flappy_bird_utils.flappy_bird import FlappyBird
import random
import json

class Bot(object):
    def __init__(self):
        # Hyper parameters
        self.qlAlpha = 0.6
        self.qlGamma = 0.8
        self.qlResolution = 10
        self.qlAliveReward = 1
        self.qlDeadReward = -1000
        self.qlEpsilon = 0.1
        self.qlExploreJumpRate = 0.1

        # Running parameters
        self.qlRound = 0  # Count the Round of Game
        self.maxScore = 0
        self.prevPipeHeight = 0
        self.currPipeHeight = 0
        self.saveCount = 25
        self.modelName = "Q1.json"

        # Q-learning parameters
        self.A = None
        self.S = None
        self.Q = {}
        self.actions = []
        self.enableQL = False

        self.load()
    def load(self):
        try:
            f = open(self.modelName, 'r')
        except IOError:
            return
        self.Q = json.load(f)
        f.close()
    def save(self):
        if self.qlRound % self.saveCount == 0:
            f = open(self.modelName, 'w')
            json.dump(self.Q, f)
            f.close()
    def move(self, state):
        S_Next = self.getQlState(state)
        if not S_Next in self.Q:
            self.Q[S_Next] = [0, 0]
        A_Next = 0
        # ε-greedy
        if (random.random() < self.qlEpsilon):
            A_Next = 1 if random.random() < self.qlExploreJumpRate else 0
        elif S_Next in self.Q:
            A_Next = 1 if self.Q[S_Next][0] < self.Q[S_Next][1] else 0
        self.actions.append([self.S, S_Next, self.A])
        self.A = A_Next
        self.S = S_Next
        return A_Next

    def getQlState(self, state):
        [playerX, playerY, playerVelY, pipeList, pipeWidth] = state
        index = 0
        for i in range(len(pipeList)):
            if pipeList[i]['x'] + pipeWidth >= playerX:
                index = i
                break
        dx = int(pipeList[index]['x'] - playerX)
        dy = int(pipeList[index]['y'] - playerY)
        if dx < 140:
            dx = dx - dx % 5
        else:
            dx = dx - dx % 30
        if dy < 180:
            dy = dy - dy % 5
        else:
            dy = dy - dy % 30
        return str(dx) + "_" + str(dy) + "_" + str(playerVelY)

    def train(self):
        self.qlRound += 1
        actions = list(reversed(self.actions))
        dead = 4
        for action in actions:
            [S, S_, A] = action
            if dead != 0:
                self.updateQ(self.Q, S, S_, A, self.qlDeadReward)
                dead -= 1
            else:
                self.updateQ(self.Q, S, S_, A, self.qlAliveReward)
        self.save()
        self.reset()

    def updateQ(self, Q, S, S_, A, R):
        if S and S_ and A in [0, 1] and S in Q and S_ in Q:
            Q[S][A] = (1 - self.qlAlpha) * Q[S][A] + self.qlAlpha * (R + self.qlGamma * max(Q[S_]))
        self.Q = Q

    def reset(self):
        self.A = None
        self.S = None
        self.actions = []
        self.currPipeHeight = 0
        self.prevPipeHeight = 0


def train_model(env, bot):
    env.frame_step(0)
    count = 0
    while True:
        bot.qlEpsilon = bot.qlEpsilon - bot.qlEpsilon * 0.01
        # if count % 2 != 0:
        #     continue
        state = env.get_q_state()
        action = bot.move(state)
        _, _, dead = env.frame_step(action)
        if(dead == True):
            bot.train()



if __name__ == "__main__":
    env = FlappyBird()
    bot = Bot()
    train_model(env, bot)
