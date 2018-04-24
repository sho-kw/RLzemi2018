import tensorflow as tf
from keras import backend as K
K.set_session(tf.Session(graph=tf.get_default_graph(), config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))

from kdrl.agents.dqn import DQNAgent
from kdrl.policy import *
from kdrl.trainer import GymTrainer
import numpy as np

from keras.models import Sequential
from keras.layers import InputLayer, Dense

import gym

EPOCHS_PER_TRAIN = 200   #PARAM
NUM_TEST = 1
EPOCHS_PER_TEST = 10

def get_model(state_shape, num_actions):#PARAM


def test(test_id):
    np.random.seed(test_id)
    tf.set_random_seed(test_id)
    env = gym.make('CartPole-v0')
    env.seed(test_id)
    #
    state_shape = env.observation_space.shape
    num_actions = env.action_space.n
    agent = DQNAgent(action_space=num_actions,
                     core_model=get_model(state_shape, num_actions),
                     optimizer='adam',#PARAM
                     policy=EpsilonGreedy(eps=0.5),#PARAM
                     loss='mean_squared_error',
                     memory=30000,
                     )
    trainer = GymTrainer(env, agent)
    # training
    trainer.train(EPOCHS_PER_TRAIN)
    # test
    result = trainer.test(EPOCHS_PER_TEST, render=True)
    return result['reward'].count(200)

def main():
    success = sum([test(i) for i in range(NUM_TEST)])
    print('result :',
          'OK ' if success > NUM_TEST*EPOCHS_PER_TEST*0.75 else 'NG ',
          '({0:.3f}%)'.format(100*success/(NUM_TEST*EPOCHS_PER_TEST)))

if __name__ == '__main__':
    main()