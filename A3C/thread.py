""" Training thread for A3C
"""

import numpy as np
from threading import Thread, Lock
from keras.utils import to_categorical
from utils.networks import tfSummary

episode = 0
lock = Lock()

def training_thread(agent, Nmax, env, action_dim, f, summary_writer, tqdm, render):
    """ Build threads to run shared computation across
    """
    global episode
    while episode < Nmax:
        # Reset episode
        time, cumul_reward, done = 0, 0, False
        old_state = env.reset()
        old_state = np.reshape(old_state, (1,) + env.observation_space.shape)
        actions, states, rewards = [], [], []
        while not done and episode < Nmax:
            if render:
                with lock: env.render()
            # Actor picks an action (following the policy)
            a = agent.policy_action(old_state)
            # Retrieve new state, reward, and whether the state is terminal
            new_state, r, done, _ = env.step(a)
            # Memorize (s, a, r) for training
            actions.append(to_categorical(a, action_dim))
            rewards.append(r)
            states.append(old_state)
            # Update current state
            old_state = np.reshape(new_state, (1,) + env.observation_space.shape)
            cumul_reward += r
            time += 1
            # Asynchronous training
            if(time%f==0 or done):
                lock.acquire()
                agent.train_models(states, actions, rewards, done)
                lock.release()
                actions, states, rewards = [], [], []

        # Export results for Tensorboard
        score = tfSummary('score', cumul_reward)
        summary_writer.add_summary(score, global_step=episode)
        summary_writer.flush()

        # textfile = open("reward_file.txt", "w")
        # print(rewards)
        # for element in list(rewards):
        #     textfile.write(element + "\n")
        # textfile.close()
        # textfile = open("states_file.txt", "w")
        # for element in list(states):
        #     textfile.write(element + "\n")
        # textfile.close()

        # Update episode count
        with lock:
            tqdm.set_description("Score: " + str(cumul_reward))
            tqdm.update(1)
            if(episode < Nmax):
                episode += 1
