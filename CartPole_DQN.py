import numpy as np
import numpy.random as nprand
from time import sleep
import math
import gym

from kdrl.agents import *
from kdrl.policy import *
from kdrl.memory import *
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Dropout


def main():
    env = gym.make('CartPole-v0')
    nprand.seed(123)
    env.seed(123)
    #env.env.theta_threshold_radians = 45 * 2 * math.pi / 360
    env.render()
    
    training = True
    

#    def update(state, action, reward, end, next_state):
#        if end:
#            val = Q[state_to_index(state)][action]
#            Q[state_to_index(state)][action] += alpha * (reward - Q[state_to_index(state)][action])
#        else:
#            Q[state_to_index(state)][action] += alpha * (reward + gamma * np.max(Q[state_to_index(next_state)]) - Q[state_to_index(state)][action])
    
#    def state_to_index(state):
#        x, dx, w, dw = state

#        def search_index(val, l):
#            idx = 0
#            for boundary in l:
#                if val > boundary:
#                    idx += 1
#                else:
#                    break
#            return idx
#            return np.digitize(np.array([val]), l)[0]
#
#        return search_index(x, X), search_index(dx, dX), search_index(w, W), search_index(dw, dW)

    def get_model(state_shape, action_space):
        return Sequential([InputLayer(input_shape=state_shape),
                        Dense(16, activation='relu'),
                        Dense(16, activation='relu'),        
                        Dense(action_space)])

    state_shape = (4,)
    action_space = 2
    agent = DQNAgent(core_model=get_model(state_shape, action_space),
                     action_space=action_space,
                     optimizer='adam',
                     policy=Boltzmann(),
                     memory=50000,
                     batch_size=32
                     )
    clear_count = 0

    for episode in range(2000 + 1):
        state = env.reset()
        done = False
        t = 0
        training = (episode % 200 != 0)
        #reward_sum = 0
        #print(episode)
        print(state)
        action = agent.start_episode(state)
        
        while not done:
            if not training:
                env.render()
                #sleep(0.01)
            #reward_sum += reward
            if not done:
                next_state, reward, done, _ = env.step(action)
                reward = 1 - abs(next_state[2])
                action = agent.step(state, reward)
                continue
            else:
                next_state, reward, done, _ = env.step(action)
                reward = 0
                agent.end_episode(state, reward)
                break
            #update(state, action, reward, done, next_state)
            state = next_state
            t += 1
            if done:
                if not training:
                    print("Episode {} finished after {} timesteps".format(episode, t + 1))
                    if t >= 200:
                        print(state, 'CLEAR')
                        clear_count += 1
                    else:
                        print(state)
                break
    
    print('Result : ', clear_count)
       
    
          
if __name__ == '__main__':
    main()