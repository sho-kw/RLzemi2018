# -*- coding: utf-8 -*-
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import numpy as np
import numpy.random as nprand
import gym
from gym import wrappers

gamma = 0.9 # PARAM
alpha = 0.4 # PARAM
eps = 0.01 # PARAM
c = 0.225 # PARAM

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

def run():
    env = gym.make("FrozenLake-v0")
    env = wrappers.Monitor(env, directory="/tmp/frozenlake-v0", force=True)
    nprand.seed(123)
    env.seed(123)
    logger.info("Observation Space: %d, Action Space: %d" % (env.observation_space.n, env.action_space.n))
    max_episode = 10000

    V = np.zeros(env.observation_space.n) # Action Values
    ACTOR = np.full((16, 4) ,0.25)

    Gs = [] # Revenues
    best = -1

    def choose_action(state):
        random = nprand.rand()
        if random < ACTOR[state][0]:
            return LEFT
        elif random < ACTOR[state][0] + ACTOR[state][1]:
            return DOWN
        elif random < ACTOR[state][0] + ACTOR[state][1] + ACTOR[state][2]:
            return RIGHT
        else:
            return UP
        
    
    def update(state, action, reward, end, next_state):
        if end:
            val = V[state]
            V[state] += alpha * (reward - V[state])
        else:

            TDerror = reward + gamma * V[next_state] - V[state]
            if TDerror >= 0:
                ACTOR[state][action] = (c + ACTOR[state][action])/(c + 1)
                for i in range(4): 
                    if i == action:
                        continue
                    ACTOR[state][i] = ACTOR[state][i]/(c + 1)
            V[state] += alpha * (TDerror)



    for episode in range(max_episode):
        x = env.reset()
        X = [] # States
        done = False
        while not done:
            if nprand.rand() < eps:
                a = nprand.randint(env.action_space.n)
            else:
                a = choose_action(x)
            X.append(x)
            x, r, done, info = env.step(a)
            if done:
                r = (2*r - 1) * 100.0
            logger.debug("State: %d, Reward: %d, Done: %s, Info: %s" % (x, r, done, info))
            x_pre = X[-1]
            update(x_pre, a, r, done, x)
        Gs.append(int(r > 0))
        avg = np.mean(Gs[-100:])
        best = max(best, avg)
        logger.info("Episode: %d, End of turn: %d, Revenue: %.2f, Average: %.2f, Best: %.2f" % (episode, len(X), r, avg, best))

if __name__ == "__main__":
    run()
