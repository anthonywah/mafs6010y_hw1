import yfinance as yf
import pandas as pd
from pytz import timezone, utc
import datetime
import logging
import _pickle as cp
import os
from contextlib import closing
import multiprocessing.pool as mp_pool


PROJECT_DIR = os.path.dirname(os.path.realpath(__file__))

CACHE_DIR = os.path.join(PROJECT_DIR, 'cache')

RESULT_IMAGE_DIR = os.path.join(PROJECT_DIR, 'result_graphs')

FORMAT_STR = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(filename)-24s:%(lineno)5s | %(message)s"


def custom_tz(*args):
    utc_dt = utc.localize(datetime.datetime.utcnow())
    my_tz = timezone('Asia/Hong_Kong')
    converted = utc_dt.astimezone(my_tz)
    return converted.timetuple()


class MyFormatter(logging.Formatter):
    def __init__(self, input_format_dict):
        super().__init__()
        self.format_dict = input_format_dict
        self.format_str = FORMAT_STR.format('', '')
        self.date_fmt = '%Y-%m-%d %H:%M:%S'

    def format(self, record):
        log_fmt = self.format_dict.get(record.levelno, self.format_str)
        formatter = logging.Formatter(log_fmt, self.date_fmt)
        return formatter.format(record)


def get_logger(name):
    color_formats = {logging.INFO: FORMAT_STR}
    logger = logging.getLogger(name)
    logging.Formatter.converter = custom_tz
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(MyFormatter(color_formats))
    logger.addHandler(ch)
    logger.propagate = False
    return logger


helper_logger = get_logger('helper')

log_msg = helper_logger.info


def save_pkl_helper(pkl, save_path):
    with open(save_path, 'wb') as f:
        cp.dump(pkl, f)
    return


def read_pkl_helper(input_path):
    with open(input_path, 'rb') as f:
        res = cp.load(f)
    return res


def get_sp_members():
    res_df = pd.read_csv('sp500.csv')
    return res_df['Symbol'].unique().tolist()


def get_sp_price(input_start_date, input_end_date):
    return get_price_data(input_stock_code='^GSPC',
                          input_start_date=input_start_date,
                          input_end_date=input_end_date)


def get_price_data(input_stock_code, input_start_date, input_end_date):
    ticker = yf.Ticker(input_stock_code)
    res_df = ticker.history(start=input_start_date,
                            end=input_end_date,
                            period='1D').reset_index()
    res_df = res_df.loc[res_df['Date'] >= input_start_date, :].reset_index(drop=True)
    res_df.rename(columns={i: i.replace(' ', '_').lower() for i in res_df.columns}, inplace=True)
    res_df.loc[:, 'date'] = res_df['date'].astype(str)
    res_df.rename(columns={'close': 'price'}, inplace=True)
    res_df = res_df[['date', 'price', 'volume']]
    log_msg(f'Retrieved {input_stock_code:<6} : {res_df["date"].min()} - {res_df["date"].max()}')
    return res_df


def get_stock_list_price(input_stock_list, input_start_date, input_end_date):
    res_dict = {}
    for one_stock in input_stock_list:
        res_dict[one_stock] = get_price_data(input_stock_code=one_stock,
                                             input_start_date=input_start_date,
                                             input_end_date=input_end_date)
    return res_dict


def get_experiment_data():
    log_msg(f'Getting experiment data')

    # Get cache if available
    cache_path = os.path.join(CACHE_DIR, 'data_cache.pkl')
    if os.path.exists(cache_path):
        log_msg(f'Reading from cache')
        return read_pkl_helper(cache_path)

    # Else Call yfinance
    target_start_date = '2011-01-01'
    target_end_date = '2021-12-31'
    kwarg = {'input_start_date': target_start_date,
             'input_end_date': target_end_date}
    res_dict = get_stock_list_price(input_stock_list=get_sp_members(), **kwarg)
    sp_df = get_sp_price(**kwarg)
    res_dict['SP500'] = sp_df
    log_msg(f'Calculating indicators')
    for k, v in res_dict.items():
        # Ensure same dates as SP500, else fill with last day price
        one_df = pd.merge(v, sp_df[['date']], on='date', how='right').reset_index(drop=True)
        if sum(one_df['price'].isna()):
            one_df.loc[:, 'price'] = one_df['price'].fillna(method='ffill')
        one_df.loc[:, 'rsi'] = rsi(one_df['price'])
        one_df.loc[:, 'zprice'] = zprice(one_df['price'])
        one_df.loc[:, 'zvol'] = zvol(one_df['volume'])
        res_dict[k] = one_df
    res_dict = {k: v.loc[v['date'] >= '2012-01-01', :].reset_index(drop=True)
                for k, v in res_dict.items()}
    save_pkl_helper(res_dict, cache_path)
    log_msg(f'Saved cache')
    return res_dict


def rsi(input_price_ps, window=14):
    """ Ref: https://school.stockcharts.com/doku.php?id=technical_indicators:relative_strength_index_rsi

    :param input_price_ps:
    :param window:
    :return:
    """
    df = pd.DataFrame(input_price_ps)
    df.columns = ['price']
    df.loc[:, 'diff'] = df['price'].diff(1)
    df.loc[:, 'gain'] = df['diff'].clip(lower=0)
    df.loc[:, 'loss'] = df['diff'].clip(upper=0).abs()
    for i in ['gain', 'loss']:
        df.loc[:, f'avg_{i}'] = df[i].rolling(window).mean()
        df.loc[:window, i] = df.loc[:window, f'avg_{i}']
        df.loc[:, f'avg_{i}'] = df[i].ewm(alpha=1/window, adjust=False).mean()
    df.loc[:, 'rs'] = df['avg_gain'] / df['avg_loss']
    df.loc[:, 'rsi'] = 100 - (100 / (1.0 + df['rs']))
    return df['rsi']


def zprice(input_price_ps, window=14):
    """

    :param input_price_ps:
    :param window:
    :return:
    """
    df = pd.DataFrame(input_price_ps)
    df.columns = ['price']
    rolling = df['price'].rolling(window)
    df.loc[:, 'zprice'] = (df['price'] - rolling.mean()) / rolling.std()
    return df['zprice']


def zvol(input_vol_ps, window=14):
    """

    :param input_vol_ps:
    :param window:
    :return:
    """
    df = pd.DataFrame(input_vol_ps)
    df.columns = ['volume']
    rolling = df['volume'].rolling(window)
    df.loc[:, 'zvol'] = (df['volume'] - rolling.mean()) / rolling.std()
    return df['zvol']


def helper_pool_run(input_func, input_arg_list):
    worker_count = int(os.cpu_count() * 4)
    with closing(mp_pool.Pool(worker_count)) as p:
        res_list = p.starmap(input_func, input_arg_list)
    return res_list
