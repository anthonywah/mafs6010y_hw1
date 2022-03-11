import timeit
import matplotlib.pyplot as plt
from data_access import *
import numpy as np
import itertools


CLUSTER_MAP = {
    'zprice<-1|zvol<=0|rsi<40':             0,
    'zprice<-1|zvol<=0|40<=rsi<=60':        1,
    'zprice<-1|zvol<=0|60<rsi':             2,
    'zprice<-1|0<zvol|rsi<40':              3,
    'zprice<-1|0<zvol|40<=rsi<=60':         4,
    'zprice<-1|0<zvol|60<rsi':              5,
    '-1<=zprice<=1|zvol<=0|rsi<40':         6,
    '-1<=zprice<=1|zvol<=0|40<=rsi<=60':    7,
    '-1<=zprice<=1|zvol<=0|60<rsi':         8,
    '-1<=zprice<=1|0<zvol|rsi<40':          9,
    '-1<=zprice<=1|0<zvol|40<=rsi<=60':     10,
    '-1<=zprice<=1|0<zvol|60<rsi':          11,
    '1<zprice|zvol<=0|rsi<40':              12,
    '1<zprice|zvol<=0|40<=rsi<=60':         13,
    '1<zprice|zvol<=0|60<rsi':              14,
    '1<zprice|0<zvol|rsi<40':               15,
    '1<zprice|0<zvol|40<=rsi<=60':          16,
    '1<zprice|0<zvol|60<rsi':               17,
}

COLORS_LIST = ['blue', 'red', 'purple', 'pink', 'green']


class StockSelectBandit:
    def __init__(self, input_epsilon, input_step_size, input_verbose=False):
        self.epsilon = input_epsilon
        self.step_size = input_step_size
        self.init_est = 0.0
        self.k = 18
        self.reward_df = calc_cluster_rewards()
        self.max_time = self.reward_df['round'].max() + 1
        self.reward_gb = self.reward_df.groupby(['round', 'C'])
        self.action_reward_seq = []
        self.cur_reward = 0.0
        self.average_reward = 0.0
        self.total_reward = 0.0
        self.indices = np.arange(self.k)
        self.q_est = np.zeros(self.k) + self.init_est
        exp_est = np.exp(self.q_est)
        self.action_prob = exp_est / np.sum(exp_est)
        self.time = 0
        self.verbose = input_verbose

    def log(self, msg):
        if self.verbose:
            log_msg(msg)

    def reset(self):
        self.action_reward_seq = []
        self.cur_reward = 0.0
        self.average_reward = 0.0
        self.total_reward = 0.0
        self.q_est = np.zeros(self.k) + self.init_est
        exp_est = np.exp(self.q_est)
        self.action_prob = exp_est / np.sum(exp_est)
        self.time = 0
        self.log(f'Reset bandit environment finished')
        return

    def action(self):

        # Exploration
        if np.random.rand() < self.epsilon:
            self.log(f't = {self.time:>3} - Exploration')
            res_choice = np.random.choice(self.k)

        # Exploitation - Gradient Bandit Algorithm
        else:
            self.log(f't = {self.time:>3} - Exploitation')
            e_est = np.exp(self.q_est)
            self.action_prob = e_est / np.sum(e_est)
            res_choice = np.random.choice(self.indices, p=self.action_prob)

        return res_choice

    def step(self, input_action):
        self.cur_reward = self.observe_reward(input_round=self.time, input_c=input_action)
        self.time += 1
        self.action_reward_seq.append((input_action, self.cur_reward))
        self.average_reward += (self.cur_reward - self.average_reward) / self.time
        self.total_reward += self.cur_reward
        rew_str = f'{self.cur_reward:.6f}'
        avg_rew_str = f'{self.average_reward:.6f}'
        total_rew_str = f'{self.total_reward:.6f}'
        self.log(f'Round {self.time - 1:>3} - Reward = {rew_str:>9} - Avg Reward = {avg_rew_str:>6} - Total Reward = {total_rew_str:>9}')

        one_hot = np.zeros(self.k)
        one_hot[input_action] = 1
        baseline = self.average_reward
        self.q_est += self.step_size * (self.cur_reward - baseline) * (one_hot - self.action_prob)
        return self.cur_reward, self.total_reward

    def observe_reward(self, input_round, input_c):
        reward = self.reward_gb.get_group((input_round, input_c))['reward'].tolist()[0]
        return reward


def simulate_one_bandit(input_bandit_num, input_bandit, input_max_time, input_sim_count):
    rewards = np.zeros((input_sim_count, input_max_time))
    total_rewards = np.zeros((input_sim_count, input_max_time))
    start_time = timeit.default_timer()
    for j in range(input_sim_count):
        input_bandit.reset()
        for k in range(input_max_time):
            action = input_bandit.action()
            reward, total_reward = input_bandit.step(input_action=action)
            rewards[j, k] = reward
            total_rewards[j, k] = total_reward
        log_msg(f'Bandit # {input_bandit_num} - Sim # {j + 1:>3} - '
                f'Avg Reward = {input_bandit.average_reward:.6f} - Total Reward = {input_bandit.total_reward:.6f}')
    log_msg(f'Bandit # {input_bandit_num} Done - time elapsed = {timeit.default_timer() - start_time:.2f}s')
    return input_bandit_num, input_bandit, rewards, total_rewards


def simulate_bandits(input_bandits, input_sim_count):
    max_time = input_bandits[0].max_time
    rewards = np.zeros((len(input_bandits), input_sim_count, max_time))
    total_rewards = np.zeros((len(input_bandits), input_sim_count, max_time))
    input_params_ls = [(i, bandit, max_time, input_sim_count) for i, bandit in enumerate(input_bandits)]
    res_ls = helper_pool_run(input_func=simulate_one_bandit, input_arg_list=input_params_ls)
    for one_res in res_ls:
        i, bandit, i_rewards, i_total_rewards = one_res
        rewards[i] = i_rewards
        total_rewards[i] = i_total_rewards
    mean_rewards = rewards.mean(axis=1)
    mean_total_rewards = total_rewards.mean(axis=1)
    return mean_rewards, mean_total_rewards


def figure_base_case():
    bandits = [StockSelectBandit(input_epsilon=0.1, input_step_size=0.1, input_verbose=False)]
    rewards, total_rewards = simulate_bandits(input_bandits=bandits, input_sim_count=1000)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    axs[0].plot(rewards[0])
    axs[0].set_title('Mean Reward - 1000 Sim', fontsize=28)
    axs[1].plot(total_rewards[0])
    axs[1].set_title('Total Reward - 1000 Sim', fontsize=28)
    for ax in axs:
        ax.grid(True)
    fig.tight_layout()
    save_path = os.path.join(RESULT_IMAGE_DIR, 'figure_base_case.png')
    plt.savefig(save_path)
    log_msg(f'Saved graph to {save_path}')
    return


def figure_alter_epsilon():
    params_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    bandits = [StockSelectBandit(input_epsilon=i, input_step_size=0.1, input_verbose=False)
               for i in params_list]
    rewards, total_rewards = simulate_bandits(input_bandits=bandits, input_sim_count=1000)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    for i in range(len(params_list)):
        axs[0].plot(rewards[i], color=COLORS_LIST[i])
        axs[0].set_title(f'Mean Reward [epsilon={params_list[i]}]- 1000 Sim', fontsize=28)
        axs[1].plot(total_rewards[i], color=COLORS_LIST[i])
        axs[1].set_title(f'Total Reward [epsilon={params_list[i]}] - 1000 Sim', fontsize=28)
    for ax in axs:
        ax.grid(True)
    fig.tight_layout()
    save_path = os.path.join(RESULT_IMAGE_DIR, 'figure_alter_epsilon.png')
    plt.savefig(save_path)
    log_msg(f'Saved graph to {save_path}')
    return


def figure_alter_step_size():
    params_list = [0.01, 0.05, 0.1, 0.2, 0.5]
    bandits = [StockSelectBandit(input_epsilon=0.1, input_step_size=i, input_verbose=False)
               for i in params_list]
    rewards, total_rewards = simulate_bandits(input_bandits=bandits, input_sim_count=1000)
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(20, 16))
    for i in range(len(params_list)):
        axs[0].plot(rewards[i], color=COLORS_LIST[i])
        axs[0].set_title(f'Mean Reward [step size={params_list[i]}]- 1000 Sim', fontsize=28)
        axs[1].plot(total_rewards[i], color=COLORS_LIST[i])
        axs[1].set_title(f'Total Reward [step size={params_list[i]}] - 1000 Sim', fontsize=28)
    for ax in axs:
        ax.grid(True)
    fig.tight_layout()
    save_path = os.path.join(RESULT_IMAGE_DIR, 'figure_alter_step_size.png')
    plt.savefig(save_path)
    log_msg(f'Saved graph to {save_path}')
    return


def calc_cluster_rewards():
    log_msg('Calculating clusters and their rewards')

    # Get .cache if available
    cache_path = os.path.join(CACHE_DIR, 'reward_cache.pkl')
    if os.path.exists(cache_path):
        log_msg(f'Reading from cache')
        return read_pkl_helper(cache_path)

    input_data_dict = get_experiment_data()
    df_ls = []
    for k, v in input_data_dict.items():
        v.loc[:, 'symbol'] = k
        v.loc[:, '2d_fwd_ret'] = v['price'].pct_change(2).shift(-2)
        if k != 'SP500':
            df_ls.append(v)
    df = pd.concat(df_ls).reset_index(drop=True)

    log_msg('Calculating clusters')
    df.loc[:, 'zprice_class'] = pd.cut(x=df['zprice'],
                                       bins=[-np.inf, -1, 1, np.inf],
                                       labels=['zprice<-1', '-1<=zprice<=1', '1<zprice'])
    df.loc[:, 'zvol_class'] = pd.cut(x=df['zvol'],
                                     bins=[-np.inf, 0, np.inf],
                                     labels=['zvol<=0', '0<zvol'])
    df.loc[:, 'rsi_class'] = pd.cut(x=df['rsi'],
                                    bins=[-np.inf, 40, 60, np.inf],
                                    labels=['rsi<40', '40<=rsi<=60', '60<rsi'])
    df.loc[:, 'cluster_label'] = df[['zprice_class', 'zvol_class', 'rsi_class']].agg('|'.join, axis=1)
    df.loc[:, 'C'] = df['cluster_label'].map(CLUSTER_MAP)

    log_msg('Calculating rewards')
    date_ls = sorted(df['date'].unique().tolist())
    grouped_date_ls = [date_ls[i:i + 3] for i in range(0, len(date_ls), 3)][:-1]
    date_map_dict = {}
    for i in grouped_date_ls:
        grouped_tag = f'{min(i)} to {max(i)}'
        date_map_dict.update({j: grouped_tag for j in i})
    df.loc[:, 'period'] = df['date'].map(date_map_dict)
    df = df.loc[~df['period'].isna(), :].reset_index(drop=True)
    df = df.groupby(['period', 'C']).apply(agg_one_period_cluster).reset_index()
    samp_df = pd.DataFrame(list(itertools.product(df['period'].unique().tolist(), df['C'].unique().tolist())), columns=['period', 'C'])
    df = pd.merge(df, samp_df, on=['period', 'C'], how='right')
    df = df.sort_values(['period', 'C']).reset_index(drop=True)
    df.loc[:, 'cluster_return'] = df['cluster_return'].fillna(0.0)
    df.loc[:, 'date'] = df['period'].str.split(' to ').apply(lambda x: x[0])
    sp_df = input_data_dict['SP500']
    sp_df.rename(columns={'2d_fwd_ret': 'sp_return'}, inplace=True)
    df = pd.merge(df, sp_df[['date', 'sp_return']], on='date', how='left').reset_index(drop=True)
    df.loc[:, 'reward'] = df['cluster_return'] - df['sp_return']
    df = df[['period', 'C', 'reward']]
    round_df = pd.DataFrame(sorted(df['period'].unique().tolist()), columns=['period'])
    round_df.loc[:, 'round'] = range(len(round(round_df)))
    df = pd.merge(df, round_df, how='left', on='period').reset_index(drop=True)
    log_msg('Finished cluster reward calculation')
    save_pkl_helper(df, cache_path)
    return df


def agg_one_period_cluster(xdf):
    reward = xdf.loc[xdf['date'] == xdf['date'], '2d_fwd_ret'].mean()
    return pd.Series({'cluster_return': reward})



