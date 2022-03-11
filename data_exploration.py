from data_access import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def sample_stock_fig1(target_stock_code='AAPL'):
    log_msg(f'{target_stock_code} - Getting stock price')
    df = get_price_data(input_stock_code=target_stock_code,
                        input_start_date='2011-01-01',
                        input_end_date='2021-12-31')

    log_msg(f'{target_stock_code} - Calculating RSI')
    df.loc[:, 'rsi'] = rsi(input_price_ps=df['price'])

    log_msg(f'{target_stock_code} - Calculating ZPRICE')
    df.loc[:, 'zprice'] = zprice(input_price_ps=df['price'])

    log_msg(f'{target_stock_code} - Calculating ZVOL')
    df.loc[:, 'zvol'] = zvol(input_vol_ps=df['volume'])

    df = df.iloc[100:700, :].reset_index(drop=True)

    log_msg(f'{target_stock_code} - Plotting graph')
    fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(30, 24))
    axs[0].plot(df['date'], df['price'], label='Price')
    locator = MaxNLocator(prune='both', nbins=7)
    axs[0].xaxis.set_major_locator(locator)
    axs[1].plot(df['rsi'], label='RSI')
    axs[2].plot(df['zprice'], label='Price Z-score')
    axs[3].plot(df['zvol'], label='Volume Z-score')
    axs[0].set_title(f'{target_stock_code} Sample Graph', fontsize=28)
    for ax in axs:
        ax.legend(fontsize=20)
        ax.grid(True)
    fig.tight_layout()
    save_path = os.path.join(RESULT_IMAGE_DIR, 'sample_stock.png')
    plt.savefig(save_path)
    log_msg(f'{target_stock_code} - Saved graph to {save_path}')
    return


def sample_indicator_dist_fig2():
    res_dict = get_experiment_data()
    d = {k: v for k, v in res_dict.items() if k != 'SP500'}
    df_ls = []
    for k, v in d.items():
        v.loc[:, 'symbol'] = k
        df_ls.append(v)
    df = pd.concat(df_ls).reset_index(drop=True)
    log_msg(f'Plotting distribution graph')
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(30, 24))

    # Price Z-score
    axs[0].hist(df['zprice'], bins=100)
    axs[0].set_title('Price Z-score', fontsize=28)
    axs[0].axvline(-1, color='red')
    axs[0].axvline(1, color='red')

    # Volume Z-score
    axs[1].hist(df['zvol'], bins=100)
    axs[1].set_title('Volume Z-score', fontsize=28)
    axs[1].axvline(0, color='red')

    # RSI
    axs[2].hist(df['rsi'], bins=100)
    axs[2].set_title('RSI', fontsize=28)
    axs[2].axvline(40, color='red')
    axs[2].axvline(60, color='red')

    for ax in axs:
        ax.grid(True)
    fig.tight_layout()
    save_path = os.path.join(RESULT_IMAGE_DIR, 'sample_distribution.png')
    plt.savefig(save_path)
    log_msg(f'Saved graph to {save_path}')
    return


if __name__ == '__main__':
    sample_stock_fig1()
    sample_indicator_dist_fig2()
