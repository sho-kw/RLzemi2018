from kdrl.agents.dqn import DQNAgent
from kdrl.policy import Boltzmann
from kdrl.trainer import GymTrainer
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense

import math
import gym

def get_model(state_shape, num_actions):
    return Sequential([InputLayer(input_shape=state_shape),
                       Dense(16, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(16, activation='relu'),
                       Dense(num_actions)])

def main():
    env = gym.make('CartPole-v0')
    env.env.theta_threshold_radians = 3 * 2 * math.pi / 360
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(action_space=num_actions,
                     core_model=get_model(state_shape, num_actions),
                     optimizer='adam',
                     policy=Boltzmann(),
                     memory=30000,
                     )
    trainer = GymTrainer(env, agent)



    # training
    result = trainer.train(200)
    # test
    result = trainer.test(5, render=True)
    for i, steps in enumerate(result['steps']):
        print('episode {}: {} steps'.format(i, steps))

if __name__ == '__main__':
    main()
