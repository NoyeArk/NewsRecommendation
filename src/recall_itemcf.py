import os
import math
import pickle
import random
import signal
import argparse
from random import shuffle
from collections import defaultdict

import multitasking
import numpy as np
import pandas as pd
from tqdm import tqdm  # 进度条

from utils import Logger, evaluate

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

random.seed(2020)

# 命令行参数
parser = argparse.ArgumentParser(description='itemcf 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'itemcf 召回，mode: {mode}')


def cal_sim(df: pd.DataFrame) -> (dict[int, dict[int, float]], dict[int, list]):
    """
    计算物品之间的相似度。

    Args:
        df (`pd.DataFrame`):
            包含用户点击数据的数据集。

    Returns:
        Tuple:
            - sim_dict (`dict[int, dict[int, float]]`):
                一个字典，键为物品ID，值为另一个字典，表示与该物品相似的物品及其相似度。
            - user_item_dict (`dict[int, list]`):
                一个字典，键为用户ID，值为该用户点击过的物品ID列表。
    """
    # 按照每个用户ID进行分组，并转化为字典：[用户ID: 点击过的物品ID]
    user_item_ = df.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(zip(user_item_['user_id'], user_item_['click_article_id']))

    # 计算每个物品出现的次数
    item_cnt = defaultdict(int)
    # 计算物品之间的相似度
    sim_dict = {}

    # 遍历所有用户和对应点击过的物品
    for _, items in tqdm(user_item_dict.items()):
        # 遍历该用户点击过的所有物品
        for i, item in enumerate(items):
            item_cnt[item] += 1  # 物品item出现次数加1
            sim_dict.setdefault(item, {})

            for j, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                sim_dict[item].setdefault(relate_item, 0)

                # 位置信息权重，考虑文章的正向顺序点击和反向顺序点击
                loc_alpha = 1.0 if j > i else 0.7
                loc_weight = loc_alpha * (0.9**(np.abs(j - i) - 1))
                sim_dict[item][relate_item] += loc_weight / math.log(1 + len(items))

    # 将每个物品相似度除掉对应物品出现次数的Log
    for i, js in tqdm(sim_dict.items()):
        for j, cij in js.items():
            sim_dict[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])

    return sim_dict, user_item_dict


@multitasking.task
def recall(df_query: pd.DataFrame,
           item_sim: dict[int, dict[int, float]],
           user_item_dict: dict[int, dict[int]],
           worker_id: int) -> None:
    """
    基于用户的历史交互和物品相似度，推荐相似物品。

    Args:
        df_query (`pd.DataFrame`):
            包含用户ID和物品ID的查询 DataFrame，格式应为两列：
            - `user_id`: 用户的唯一标识符
            - `item_id`: 当前查询的物品ID（-1 表示无特定物品）

        item_sim (`dict[int, dict[int, float]]`):
            物品相似度字典，键为物品ID，值为另一个字典，表示与该物品相似的其他物品及其相似度。

        user_item_dict (`dict[int, List[int]]`):
            用户与其交互过的物品字典，键为用户ID，值为该用户点击过的物品ID列表。

        worker_id (int):
            当前工作线程的唯一标识符，用于保存结果文件。
    """
    data_list = []

    # 遍历查询 DataFrame 中的每一个用户和物品ID
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)  # 初始化字典，用于存储推荐物品及其得分

        # 如果用户ID不在用户物品字典中，则跳过该用户
        if user_id not in user_item_dict:
            continue

        # 获取当前用户交互过的物品，保留最新的两个物品
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[::-1][:2]

        # 遍历用户交互过的物品
        for loc, item in enumerate(interacted_items):
            # 获取与当前物品相似的物品，按相似度排序并取前200个
            for relate_item, cij in sorted(item_sim[item].items(),
                                           key=lambda d: d[1],
                                           reverse=True)[0:200]:
                # 如果相关物品已被用户交互过，则跳过
                if relate_item not in interacted_items:
                    # 初始化推荐物品的得分
                    rank.setdefault(relate_item, 0)
                    # 根据物品位置加权累加相似度得分
                    rank[relate_item] += cij * (0.7**loc)

        # 对推荐物品进行排序，按得分选择前100个
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:100]
        item_ids = [item[0] for item in sim_items]  # 提取推荐物品的ID
        item_sim_scores = [item[1] for item in sim_items]  # 提取相似度得分

        # 创建要给临时 DataFrame 存储推荐结果
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids  # 推荐物品ID
        df_temp['sim_score'] = item_sim_scores  # 物品相似度得分
        df_temp['user_id'] = user_id  # 当前用户ID

        # 如果 item_id 为 -1，表示无特定物品，则 label 为 NaN
        if item_id == -1:
            df_temp['label'] = np.nan
        else:
            df_temp['label'] = 0  # 初始化 label 为 0
            # 如果推荐物品是当前 item_id，则将 label 设为 1
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 重新排列 DataFrame 列的顺序
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')
        # 将临时 DataFrame 添加到数据列表中
        data_list.append(df_temp)

    # 将所有临时 DataFrame 添加到数据列表中
    df_data = pd.concat(data_list, sort=False)

    # 创建保存推荐结果的目录（如果不存在）
    os.makedirs('../user_data/tmp/itemcf', exist_ok=True)
    # 将最终结果保存为 pickle 文件
    df_data.to_pickle(f'../user_data/tmp/itemcf/{worker_id}.pkl')


if __name__ == '__main__':
    if mode == 'valid':
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/sim/offline', exist_ok=True)
        sim_pkl_file = '../user_data/sim/offline/itemcf_sim.pkl'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/sim/online', exist_ok=True)
        sim_pkl_file = '../user_data/sim/online/itemcf_sim.pkl'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    item_sim, user_item_dict = cal_sim(df_click)
    f = open(sim_pkl_file, 'wb')
    pickle.dump(item_sim, f)
    f.close()

    # 召回
    n_split = max_threads
    all_users = df_query['user_id'].unique()
    shuffle(all_users)
    total = len(all_users)
    n_len = total // n_split

    # 清空临时文件夹
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]
        df_temp = df_query[df_query['user_id'].isin(part_users)]
        recall(df_temp, item_sim, user_item_dict, i)

    multitasking.wait_for_tasks()
    log.info('合并任务')

    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/itemcf'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 必须加，对其进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')

        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()

        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'itemcf: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )
    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_itemcf.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_itemcf.pkl')