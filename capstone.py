import random
import gym.spaces
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import time
from pathlib import Path
import os
from graph_utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow informational messages
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Force CPU vs GPU as this runs faster on CPU

#  Suppress warning message
#  https://github.com/openai/gym/issues/868
gym.logger.set_level(40)


#  Memory class that stores the memory buffer utilized during the replay operation
class Memory:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, b):
        batch = np.random.choice(np.arange(len(self.buffer)), size=b, replace=False)
        return [self.buffer[x] for x in batch]

    def __len__(self):
        return len(self.buffer)


#  Deep Q-Learning class that creates the DQN model.
class DQN:
    def __init__(self, state_size, lr=.001):
        self.state_size = state_size
        self.learning_rate = lr
        self.model = None

        self.action_dict = dict()
        self.reverse_actions = dict()

        key = 0

        #  Create the dictionary of input values and output conversions by creating
        #  a discrete set of values from the continuous values.
        for i in [-1.0, 0.0, 1.0]:
            for j in [-1.0, 0.0, 1.0]:
                self.action_dict[key] = np.array([i, j])
                self.reverse_actions[i, j] = key
                key += 1

        self.action_size = len(self.action_dict)

    #  Builds the DQN model given a set of parameters.  The parameters are limited to two
    #  hidden layers
    def build_model(self, params: dict):
        self.model = Sequential()
        self.model.add(Dense(params['hidden1'], input_dim=self.state_size, activation='relu'))
        if params['hidden2'] > 0:
            self.model.add(Dense(params['hidden2'], activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        if params['summary']:
            print(self.model.summary())

    # The action to take based on a given epsilon value.  If the epsilon value is greater than or
    # equal to a random value then a random action is performed; otherwise, the predicted value
    # based on the model is selected and returned.
    def act(self, s, e):
        if np.random.rand() <= e:
            r = random.randrange(self.action_size)
            return self.action_dict[r]
        act_values = self.model.predict(s)
        r = np.argmax(act_values[0])
        return self.action_dict[r]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


#  The main function that runs the testing code.
def run_algorithm(params: dict):
    env = gym.make('LunarLanderContinuous-v2')
    state_size = env.observation_space.shape[0]
    agent = DQN(state_size, lr=params['learning_rate'])
    target_agent = DQN(state_size, lr=params['learning_rate'])
    agent.build_model(params['model'])
    target_agent.build_model(params['model'])
    memory = Memory(max_size=params['buffer_size'])
    epsilon = params['epsilon']
    epsilon_stop = params['epsilon_stop']
    epsilon_decay = params['epsilon_decay']
    tau = params['tau']
    gamma = params['gamma']
    h5_name = params['h5_name']
    batch_size = params['batch_size']
    episodes = params['episodes']
    steps = params['steps']
    episode_average = params['episode_average']
    model_filename = Path(params['model_filename'])

    rewards_list = []
    episode_times = []
    reward_averages = []
    step_list = []
    buffer_growth = []
    epsilon_list = []
    episode_counter = 0
    consec_success = 5
    reward_value = 200
    reward_streak = 0
    verbose_mod = 10

    if params['load_model']:
        # Load previous agent data if exists
        if model_filename.is_file():
            agent.load(model_filename)

    total_start_time = time.time()
    for episode in range(episodes):
        epsilon_list.append(epsilon)
        episode_counter += 1
        episode_time = time.time()
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        t = 0
        for t in range(steps):
            if params['render']:
                env.render()

            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            # Translate the output into the corresponding dictionary value
            action = agent.reverse_actions[action[0], action[1]]
            memory.add((state, action, reward, next_state, done))
            state = next_state
            if done:
                break

            if t > batch_size:
                minibatch = memory.sample(batch_size)
                for s1, a, r, s2, done in minibatch:
                    target = r
                    if not done:
                        target = (r + gamma * np.amax(target_agent.model.predict(s2)[0]))
                    target_f = target_agent.model.predict(s1)
                    target_f[0][a] = target
                    agent.model.fit(s1, target_f, epochs=1, verbose=0)

                online_weights = np.array(agent.model.get_weights())
                target_weights = np.array(target_agent.model.get_weights())
                update_weights = (online_weights * tau) + ((1 - tau) * target_weights)
                target_agent.model.set_weights(update_weights)

        buffer_len = len(memory.buffer)
        buffer_growth.append(buffer_len)
        step_list.append(t)
        rewards_list.append(total_reward)
        e_time = time.time() - episode_time
        episode_times.append(e_time)
        r = 0.0
        if len(rewards_list) >= episode_average:
            r = sum(rewards_list[-episode_average:]) / episode_average
            reward_averages.append(r)
            if params['use_threshold']:
                if r >= reward_value:
                    if reward_streak >= consec_success:
                        agent.save(model_filename)
                        break
                    else:
                        reward_streak += 1
                else:
                    reward_streak = 0

        epsilon = epsilon_stop + (epsilon - epsilon_stop) * np.exp(-epsilon_decay * episode)
        if episode % verbose_mod == 0:
            if params['verbose']:
                e_sum = sum(episode_times[-verbose_mod:])
                e_time = e_sum / verbose_mod
                avg_steps = sum(step_list[-verbose_mod:]) / verbose_mod
                avg_reward = sum(rewards_list[-verbose_mod:]) / verbose_mod
                print("episode: {}/{}, avg reward: {}, threshold reward: {}, e: {:.2}, total time: {:.2}, "
                      "avg time: {:.2}, avg steps: {}, buffer: {}".
                      format(episode, episodes, int(avg_reward), int(r), epsilon, e_sum,
                             e_time, int(avg_steps), buffer_len))
            agent.save("./saved_models/" + h5_name + "_dqn_" + str(episode) + ".h5")

    if params['verbose']:
        print("total time: {}".format(time.time()-total_start_time))

    return_dict = {
        'epsilon_list': np.array(epsilon_list).T,
        'rewards_list': np.array(rewards_list).T,
        'episode_times': np.array(episode_times).T,
        'step_list': np.array(step_list).T,
        'buffer_growth': np.array(buffer_growth).T,
        'reward_averages': np.array(reward_averages).T,
        'episode_averages': np.arange(episode_average, len(reward_averages) + episode_average),
        'episode_counter': np.arange(1, episode_counter + 1)
    }

    return return_dict


#  The function runs 100 iterations of a trained model against the lunar lander
#  environment.  The results are recorded and graphs are created to show against
#  the projects metric.
def run_100(params: dict):
    env = gym.make('LunarLanderContinuous-v2')
    state_size = env.observation_space.shape[0]

    agent = DQN(state_size)
    agent.build_model(params['model'])

    episodes = params['episodes']
    steps = params['steps']
    model_filename = Path(params['model_filename'])
    graph_name = params['graph_name']

    rewards_list = []
    episode_times = []
    step_list = []
    episode_counter = 0

    if model_filename.is_file():
            agent.load(model_filename)

    total_start_time = time.time()
    for episode in range(episodes):
        episode_counter += 1
        episode_time = time.time()
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0
        t = 0
        for t in range(steps):
            env.render()
            action = agent.act(state, 0)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            next_state = np.reshape(next_state, [1, state_size])
            state = next_state
            if done:
                break

        step_list.append(t)
        rewards_list.append(total_reward)
        e_time = time.time() - episode_time
        episode_times.append(e_time)

        if params['verbose']:
            print("episode: {}/{}, score: {}, time: {:.2}, steps: {}".
                  format(episode, episodes, round(total_reward), e_time, t))

    if params['verbose']:
        print("total time: {}".format(time.time()-total_start_time))
        print("Average reward: {}".format(sum(rewards_list) / float(len(rewards_list))))

    # Create graphs
    params = {
        'x': np.array(rewards_list).T,
        'y': np.arange(1, episode_counter + 1),
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward On Trained Model',
        'filename': graph_name + '_total_reward_over_trained.png',
        'smoothed': False
    }
    single_bar_graph(params)


#  The method runs an assortment of metric tests: epsilon decay, epsilon,
#  gamma, learning_rate, and tau.  The metrics are then graphed for
#  use later.
def run_multiple_variations(settings: dict):

    decay = [0.01, 0.001, 0.0001]
    epsilon = [0.99, 0.90, 0.70]
    gamma = [0.7, 0.9, 0.99]
    learning_rate = [0.01, 0.001, 0.0001]
    tau = [1.0, 0.1, 0.01]
    colors = ['blue', 'red', 'green', 'black', 'orange', 'yellow']
    buffer_size = settings['buffer_size']
    batch_size = settings['batch_size']
    episodes = settings['episodes']
    steps = settings['steps']
    episode_average = settings['episode_average']
    static_epsilon = settings['epsilon']
    static_stop = settings['epsilon_stop']
    static_decay = settings['epsilon_decay']
    static_tau = settings['tau']
    static_gamma = settings['gamma']
    static_lr = settings['learning_rate']
    h5_name = settings['h5_name']
    render = settings['render']
    layers = settings['model']
    graph_values = {}
    values = {}

    print('')
    print('=' * 50)
    for d in range(len(decay)):
        params = {
            'buffer_size': buffer_size,
            'epsilon': static_epsilon,
            'epsilon_stop': static_stop,
            'epsilon_decay': decay[d],
            'tau': static_tau,
            'gamma': static_gamma,
            'model': {'hidden1': layers['hidden1'], 'hidden2': layers['hidden2'], 'summary': layers['summary']},
            'learning_rate': static_lr,
            'batch_size': batch_size,
            'episodes': episodes,
            'steps': steps,
            'episode_average': episode_average,
            'load_model': False,
            'model_filename': './project2_decay_dqn.h5',
            'verbose': True,
            'use_threshold': False,
            'render': render,
            'h5_name': h5_name
        }
        print('Decay: ', decay[d])
        print('-' * 25)
        values = run_algorithm(params)
        graph_values[d] = {'data': values['rewards_list'], 'color': colors[d], 'label': str(decay[d])}

    params = {
        'x': graph_values,
        'y': values['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward With Varying Epsilon Decay',
        'filename': h5_name + '_reward_per_decay.png',
        'smoothed': False
    }
    multiple_graph(params)

    graph_values = {}
    values = {}

    print('')
    print('=' * 50)
    for d in range(len(epsilon)):
        params = {
            'buffer_size': buffer_size,
            'epsilon': epsilon[d],
            'epsilon_stop': static_stop,
            'epsilon_decay': static_decay,
            'tau': static_tau,
            'gamma': static_gamma,
            'model': {'hidden1': layers['hidden1'], 'hidden2': layers['hidden2'], 'summary': layers['summary']},
            'learning_rate': static_lr,
            'batch_size': batch_size,
            'episodes': episodes,
            'steps': steps,
            'episode_average': episode_average,
            'load_model': False,
            'model_filename': './project2_epsilon_dqn.h5',
            'verbose': True,
            'use_threshold': False,
            'render': render,
            'h5_name': h5_name
        }
        print('Epsilon: ', epsilon[d])
        print('-' * 25)
        values = run_algorithm(params)
        graph_values[d] = {'data': values['rewards_list'], 'color': colors[d], 'label': str(epsilon[d])}

    params = {
        'x': graph_values,
        'y': values['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward With Varying Epsilon',
        'filename': h5_name + '_reward_per_epsilon.png',
        'smoothed': False
    }
    multiple_graph(params)

    graph_values = {}
    values = {}

    print('')
    print('=' * 50)
    for d in range(len(gamma)):
        params = {
            'buffer_size': buffer_size,
            'epsilon': static_epsilon,
            'epsilon_stop': static_stop,
            'epsilon_decay': static_decay,
            'tau': static_tau,
            'gamma': gamma[d],
            'model': {'hidden1': layers['hidden1'], 'hidden2': layers['hidden2'], 'summary': layers['summary']},
            'learning_rate': static_lr,
            'batch_size': batch_size,
            'episodes': episodes,
            'steps': steps,
            'episode_average': episode_average,
            'load_model': False,
            'model_filename': './project2_gamma_dqn.h5',
            'verbose': True,
            'use_threshold': False,
            'render': render,
            'h5_name': h5_name
        }
        print('Gamma: ', gamma[d])
        print('-' * 25)
        values = run_algorithm(params)
        graph_values[d] = {'data': values['rewards_list'], 'color': colors[d], 'label': str(gamma[d])}

    params = {
        'x': graph_values,
        'y': values['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward With Varying Gamma',
        'filename': h5_name + '_reward_per_gamma.png',
        'smoothed': False
    }
    multiple_graph(params)

    graph_values = {}
    values = {}

    print('')
    print('=' * 50)
    for d in range(len(learning_rate)):
        params = {
            'buffer_size': buffer_size,
            'epsilon': static_epsilon,
            'epsilon_stop': static_stop,
            'epsilon_decay': static_decay,
            'tau': static_tau,
            'gamma': static_gamma,
            'model': {'hidden1': layers['hidden1'], 'hidden2': layers['hidden2'], 'summary': layers['summary']},
            'learning_rate': learning_rate[d],
            'batch_size': batch_size,
            'episodes': episodes,
            'steps': steps,
            'episode_average': episode_average,
            'load_model': False,
            'model_filename': './project2_lr_dqn.h5',
            'verbose': True,
            'use_threshold': False,
            'render': render,
            'h5_name': h5_name
        }
        print('Learning Rate: ', learning_rate[d])
        print('-' * 25)
        values = run_algorithm(params)
        graph_values[d] = {'data': values['rewards_list'], 'color': colors[d], 'label': str(learning_rate[d])}

    params = {
        'x': graph_values,
        'y': values['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward With Varying Learning Rates',
        'filename': h5_name + '_reward_per_learning_rate.png',
        'smoothed': False
    }
    multiple_graph(params)

    graph_values = {}
    values = {}

    print('')
    print('=' * 50)
    for d in range(len(tau)):
        params = {
            'buffer_size': buffer_size,
            'epsilon': static_epsilon,
            'epsilon_stop': static_stop,
            'epsilon_decay': static_decay,
            'tau': tau[d],
            'gamma': static_gamma,
            'model': {'hidden1': layers['hidden1'], 'hidden2': layers['hidden2'], 'summary': layers['summary']},
            'learning_rate': static_lr,
            'batch_size': batch_size,
            'episodes': episodes,
            'steps': steps,
            'episode_average': episode_average,
            'load_model': False,
            'model_filename': './project2_tau_dqn.h5',
            'verbose': True,
            'use_threshold': False,
            'render': render,
            'h5_name': h5_name
        }
        print('Tau: ', tau[d])
        print('-' * 25)
        values = run_algorithm(params)
        graph_values[d] = {'data': values['rewards_list'], 'color': colors[d], 'label': str(tau[d])}

    params = {
        'x': graph_values,
        'y': values['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward With Varying Tau',
        'filename': h5_name + '_reward_per_tau.png',
        'smoothed': False
    }
    multiple_graph(params)


if __name__ == "__main__":

    #  Toggle on and off (True/False) the various tests below.  Running the multiple tests
    #  could take days to finish while the last four tests could still take 2-6 hours
    #  depending on hardware.
    run_benchmark_multiple_tests = False
    run_solution_multiple_tests = False
    run_benchmark_test = False
    run_solution_test = False
    run_benchmark_100_test = True
    run_solution_100_test = True

    # Creates multiple graphs of various metrics for the benchmark model
    if run_benchmark_multiple_tests:
        params = {
            'buffer_size': 50000,
            'epsilon': 0.99,
            'epsilon_stop': 0.01,
            'epsilon_decay': 0.0001,
            'tau': 0.01,
            'gamma': 0.99,
            'model': {'hidden1': 128, 'hidden2': 0, 'summary': False},
            'learning_rate': 0.001,
            'batch_size': 16,
            'episodes': 1000,
            'steps': 500,
            'episode_average': 100,
            'load_model': False,
            'model_filename': './benchmark_dqn.h5',
            'verbose': True,
            'graph_it': True,
            'use_threshold': False,
            'render': False,
            'h5_name': 'benchmark'
        }
        run_multiple_variations(params)

    # Creates multiple graphs of various metrics for the solution model
    if run_solution_multiple_tests:
        params = {
            'buffer_size': 50000,
            'epsilon': 0.99,
            'epsilon_stop': 0.01,
            'epsilon_decay': 0.0001,
            'tau': 0.01,
            'gamma': 0.99,
            'model': {'hidden1': 128, 'hidden2': 256, 'summary': False},
            'learning_rate': 0.001,
            'batch_size': 16,
            'episodes': 1000,
            'steps': 500,
            'episode_average': 100,
            'load_model': False,
            'model_filename': './solution_dqn.h5',
            'verbose': True,
            'graph_it': True,
            'use_threshold': False,
            'render': False,
            'h5_name': 'solution'
        }
        run_multiple_variations(params)

    # Runs the benchmark test with specified values found from researching the metric graph results
    if run_benchmark_test:
        params = {
            'buffer_size': 50000,
            'epsilon': 0.7,
            'epsilon_stop': 0.01,
            'epsilon_decay': 0.0001,
            'tau': 0.01,
            'gamma': 0.99,
            'model': {'hidden1': 128, 'hidden2': 0, 'summary': True},
            'learning_rate': 0.001,  # adam optimizer default is 0.001
            'batch_size': 16,
            'episodes': 5000,
            'steps': 1000,
            'episode_average': 100,
            'load_model': False,
            'model_filename': './benchmark_dqn.h5',
            'verbose': True,
            'graph_it': True,
            'use_threshold': True,
            'render': True,
            'h5_name': 'benchmark'
        }
        values = run_algorithm(params)
        graph_metrics(values, 'benchmark')

    # Runs the solution test with specified values found from researching the metric graph results
    if run_solution_test:
        params = {
            'buffer_size': 50000,
            'epsilon': 0.99,
            'epsilon_stop': 0.01,
            'epsilon_decay': 0.0001,
            'tau': 0.01,
            'gamma': 0.99,
            'model': {'hidden1': 128, 'hidden2': 256, 'summary': True},
            'learning_rate': 0.0001,
            'batch_size': 24,
            'episodes': 5000,
            'steps': 1000,
            'episode_average': 100,
            'load_model': False,
            'model_filename': './solution_dqn.h5',
            'verbose': True,
            'graph_it': True,
            'use_threshold': True,
            'render': True,
            'h5_name': 'solution'
        }
        values = run_algorithm(params)
        graph_metrics(values, 'solution')

    # Run the learned model against 100 iterations and output the benchmark's results
    if run_benchmark_100_test:
        params = {
            'model': {'hidden1': 128, 'hidden2': 0, 'summary': True},
            'episodes': 100,
            'steps': 1000,
            'model_filename': './benchmark_dqn.h5',
            'graph_name': 'benchmark',
            'verbose': True
        }
        run_100(params)

    # Run the learned model against 100 iterations and output the solution's results
    if run_solution_100_test:
        params = {
            'model': {'hidden1': 128, 'hidden2': 256, 'summary': True},
            'episodes': 100,
            'steps': 1000,
            'model_filename': './solution_dqn.h5',
            'graph_name': 'solution',
            'verbose': True
        }
        run_100(params)