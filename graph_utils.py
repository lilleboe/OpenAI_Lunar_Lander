import numpy as np
import matplotlib.pyplot as plt

#  A select of graph functions to output various metric visualizations for the capstone project


def running_mean(x, n):
    cs = np.cumsum(np.insert(x, 0, 0))
    rm = (cs[n:] - cs[:-n]) / n
    return rm


def single_graph(params: dict):
    x = params['x']
    y = params['y']
    smoothed_x = running_mean(x, 10)
    if params['smoothed']:
        print(smoothed_x)
        plt.plot(y[-len(smoothed_x):], smoothed_x, label='Moving Average', color='red')
    plt.plot(y, x, color='blue', alpha=0.5)
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.title(params['title'])
    plt.grid()
    plt.legend()
    plt.savefig(params['filename'])
    plt.close()


def single_bar_graph(params: dict):
    x = params['x']
    y = params['y']
    smoothed_x = running_mean(x, 20)
    if params['smoothed']:
        plt.plot(y[-len(smoothed_x):], smoothed_x, label='Moving Average', color='red')
    plt.bar(y, x, color='blue', alpha=.5)
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.title(params['title'])
    plt.grid()
    plt.legend()
    plt.savefig(params['filename'])
    plt.close()


def multiple_graph(params: dict):
    y = params['y']
    for key, value in params['x'].items():
        smoothed_x = running_mean(value['data'], 25)
        plt.plot(y[-len(smoothed_x):], smoothed_x, color=value['color'], label=value['label'])
    plt.xlabel(params['xlabel'])
    plt.ylabel(params['ylabel'])
    plt.title(params['title'])
    plt.grid()
    plt.legend()
    plt.savefig(params['filename'])
    plt.close()


#  Graphs a set of metrics for a model.  The graphs show rewards, times, buffer growth and steps
def graph_metrics(settings: dict, image_name):

    # Create graphs
    params = {
        'x': settings['rewards_list'],
        'y': settings['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Total Reward',
        'title': 'Total Reward Per Episode',
        'filename': 'total_reward_' + image_name + '.png',
        'smoothed': True
    }
    single_graph(params)

    params = {
        'x': settings['reward_averages'],
        'y': settings['episode_averages'],
        'xlabel': 'Episode',
        'ylabel': 'Average Rewards',
        'title': 'Average Reward Per 100 Episodes',
        'filename': 'average_rewards_over_100_' + image_name + '.png',
        'smoothed': True
    }
    single_graph(params)

    params = {
        'x': settings['episode_times'],
        'y': settings['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Time (s)',
        'title': 'Time Per Episode',
        'filename': 'time_per_episode_' + image_name + '.png',
        'smoothed': True
    }
    single_graph(params)

    params = {
        'x': settings['buffer_growth'],
        'y': settings['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Replay Memory',
        'title': 'Reply Memory Growth Per Episode',
        'filename': 'replay_memory_growth_' + image_name + '.png',
        'smoothed': False
    }
    single_graph(params)

    params = {
        'x': settings['step_list'],
        'y': settings['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Steps',
        'title': 'Steps Per Episode',
        'filename': 'steps_' + image_name + '.png',
        'smoothed': True
    }
    single_graph(params)

    params = {
        'x': settings['epsilon_list'],
        'y': settings['episode_counter'],
        'xlabel': 'Episode',
        'ylabel': 'Epsilon',
        'title': 'Epsilon Values Per Episode',
        'filename': 'epsilon_' + image_name + '.png',
        'smoothed': False
    }
    single_graph(params)

