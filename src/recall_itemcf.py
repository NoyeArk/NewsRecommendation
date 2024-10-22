import os
import argparse
import pandas as pd
from collections import defaultdict

# 命令行参数
parser = argparse.ArgumentParser(description='ItemCF召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')


def cal_sim(df: pd.DataFrame) -> (dict, dict):
    user_item_ = df.groupby('user_id')['click_article_id'].agg(lambda x: list(x)).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    item_cnt = defaultdict(int)



if __name__ == '__main__':
    # 得到命令行参数
    args = parser.parse_args()
    if args.model == 'valid':  # 线下训练
        df_click = pd.read_pickle('../data/offline/click.pkl')
        df_query = pd.read_pickle('../data/offline/query.pkl')

        os.makedirs('../data/sim/offline', exist_ok=True)
        sim_pkl_file = '../data/sim/offline/itemcf_sim.pkl'
    else:  # 线上测试
        df_click = pd.read_pickle('../data/online/click.pkl')
        df_query = pd.read_pickle('../data/online/query.pkl')

        os.makedirs('../data/sim/online', exist_ok=True)
        sim_pkl_file = '../data/sim/online/itemcf_sim.pkl'

    item_sim, user_item_dict = cal_sim(df_click)
