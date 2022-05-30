import random
import numpy as np

from tqdm import tqdm
from keras.models import Model
from keras import regularizers
from keras.utils import to_categorical
from keras.layers import Input, Dense, Flatten

from .critic import Critic
from .actor import Actor
from utils.networks import tfSummary
from utils.stats import gather_stats

class A2C:
    """ Actor-Critic Main Algorithm
    """

    def __init__(self, act_dim, env_dim, k, gamma = 0.99, lr = 0.0001):
        """ Initialization
        """
        # Environment and A2C parameters
        self.act_dim = act_dim
        self.env_dim = (k,) + env_dim
        self.gamma = gamma
        self.lr = lr
        # Create actor and critic networks
        self.shared = self.buildNetwork()
        self.actor = Actor(self.env_dim, act_dim, self.shared, lr)
        self.critic = Critic(self.env_dim, act_dim, self.shared, lr)
        # Build optimizers
        self.a_opt = self.actor.optimizer()
        self.c_opt = self.critic.optimizer()

    def buildNetwork(self):
        """ Assemble shared layers
        """
        inp = Input((self.env_dim))
        x = Flatten()(inp)
        x = Dense(64, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(inp, x)

    def policy_action(self, s):
        """ Use the actor to predict the next action to take, using the policy
        """
        return np.random.choice(np.arange(self.act_dim), 1, p=self.actor.predict(s).ravel())[0]


    def discount(self, r):
        """ Compute the gamma-discounted rewards over an episode
        """
        discounted_r, cumul_r = np.zeros_like(r), 0
        for t in reversed(range(0, len(r))):
            cumul_r = r[t] + cumul_r * self.gamma
            discounted_r[t] = cumul_r
        return discounted_r

    def train_models(self, states, actions, rewards, done):
        """ Update actor and critic networks from experience
        """
        # Compute discounted rewards and Advantage (TD. Error)
        discounted_rewards = self.discount(rewards)
        state_values=[]
        for i in range(len(states)):
            tmp = self.critic.predict(np.array(states[i]))
            state_values.append(tmp)
        advantages = discounted_rewards - np.reshape(np.array(state_values), (1,np.product(np.array(state_values).shape)))[0]
        final_states = []
        for i in range(len(states)):
            tmp = np.repeat(states[i], 4, axis=0)
            final_states.append(tmp)

        # Networks optimization
        self.a_opt([final_states, actions, advantages])
        self.c_opt([final_states, discounted_rewards])

    def train(self, env, args, summary_writer):
        """ Main A2C Training Algorithm
        """

        results = []

        # Main Loop
        tqdm_e = tqdm(range(args.nb_episodes), desc='Score', leave=True, unit=" episodes")
        for e in tqdm_e:

            # Reset episode
            time, cumul_reward, done = 0, 0, False
            old_state = env.reset()
            old_state = np.reshape(old_state, (1,) + env.observation_space.shape)
            actions, states, rewards = [], [], []

            while not done:
                # print(old_state)
                if args.render: env.render()
                # Actor picks an action (following the policy)
                # print("111111111111111111111111111111111")
                a = self.policy_action(old_state)
                # print("222222222222222222222222222222222")
                # Retrieve new state, reward, and whether the state is terminal
                new_state, r, done, _ = env.step(a)
                # Memorize (s, a, r) for training
                actions.append(to_categorical(a, self.act_dim))
                rewards.append(r)
                states.append(old_state)
                # Update current state
                old_state = np.reshape(new_state, (1,) + env.observation_space.shape)
                cumul_reward += r
                time += 1
            # Train using discounted rewards ie. compute updates
            self.train_models(states, actions, rewards, done)

            # Gather stats every episode for plotting
            if(args.gather_stats):
                mean, stdev = gather_stats(self, env)
                results.append([e, mean, stdev])

            # Export results for Tensorboard
            score = tfSummary('score', cumul_reward)
            summary_writer.add_summary(score, global_step=e)
            summary_writer.flush()

            # Display score
            tqdm_e.set_description("Score: " + str(cumul_reward))
            tqdm_e.refresh()
            
        textfile = open("reward_file.txt", "w")
        for element in rewards:
            textfile.write(element + "\n")
        textfile.close()
        textfile = open("states_file.txt", "w")
        for element in states:
            textfile.write(element + "\n")
        textfile.close()
        return results

    def save_weights(self, path):
        path += '_LR_{}'.format(self.lr)
        self.actor.save(path)
        self.critic.save(path)

    def load_weights(self, path_actor, path_critic):
        self.critic.load_weights(path_critic)
        self.actor.load_weights(path_actor)
