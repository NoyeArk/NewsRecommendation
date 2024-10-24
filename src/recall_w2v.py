import argparse
import math
import os
import pickle
import random
import signal
import warnings
from collections import defaultdict
from random import shuffle

import multitasking
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

from utils import Logger, evaluate

warnings.filterwarnings('ignore')

max_threads = multitasking.config['CPU_CORES']
multitasking.set_max_threads(max_threads)
multitasking.set_engine('process')
signal.signal(signal.SIGINT, multitasking.killall)

seed = 2020
random.seed(seed)

# 命令行参数
parser = argparse.ArgumentParser(description='w2v 召回')
parser.add_argument('--mode', default='valid')
parser.add_argument('--logfile', default='test.log')

args = parser.parse_args()

mode = args.mode
logfile = args.logfile

# 初始化日志
os.makedirs('../user_data/log', exist_ok=True)
log = Logger(f'../user_data/log/{logfile}').logger
log.info(f'w2v 召回，mode: {mode}')


def word2vec(df_, f1, f2, model_path):
    """
    训练 Word2Vec 模型并生成物品向量映射。

    Args:
        df_ (`pd.DataFrame`):
            输入的 DataFrame，包含用于训练模型的文本数据。
        f1 (`str`):
            用于分组的列名，通常为用户ID列。
        f2 (`str`):
            用于生成句子的列名，通常为物品ID列。
        model_path (`str`):
            存储训练好的 Word2Vec 模型的路径。

    Returns:
        Dict[int, np.ndarray]:
            一个字典，键为物品ID（整数），值为对应的物品向量（NumPy 数组）。
    """
    df = df_.copy()
    # 按照 f1 列进行分组，agg 生成一个列表，列名为 f1_f2_list
    tmp = df.groupby(f1, as_index=False)[f2].agg(
        {'{}_{}_list'.format(f1, f2): list})

    # 提取物品ID列的值并转化为列表
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    del tmp['{}_{}_list'.format(f1, f2)]  # 删除临时列

    words = []  # 存储所有单词
    for i in range(len(sentences)):
        x = [str(x) for x in sentences[i]]  # 将每个物品转换为字符串
        sentences[i] = x  # 更新句子列表
        words += x  # 添加到打算那次列表

    # 检查模型是否已存在
    if os.path.exists(f'{model_path}/w2v.m'):
        model = Word2Vec.load(f'{model_path}/w2v.m')  # 加载已有模型
    else:
        # 创建新的 Word2Vec 模型
        model = Word2Vec(sentences=sentences,
                         vector_size=256,  # 向量维度
                         window=3,         # 上下文窗口大小
                         min_count=1,      # 最小词频
                         sg=1,             # 使用 Skip-gram 模型
                         hs=0,             # 不适用层次软最小化
                         seed=seed,        # 随机种子
                         negative=5,       # 负采样数量
                         workers=10,       # 并行工作线程数
                         epochs=1)         # 训练的迭代轮数
        model.save(f'{model_path}/w2v.m')  # 保存模型

    # 初始化物品向量映射字典
    article_vec_map = {}
    for word in set(words):  # 遍历唯一单词
        if word in model.wv:  # 检查单词是否在模型中
            article_vec_map[int(word)] = model.wv[word]  # 将物品ID和向量映射

    # 返回物品向量映射
    return article_vec_map


@multitasking.task
def recall(df_query, article_vec_map, article_index, user_item_dict, worker_id):
    """
    根据用户的历史行为和文章向量，进行推荐并生成相应的推荐结果。

    Args:
        df_query (`pd.DataFrame`):
            查询数据，包含用户ID和目标文章ID。
        article_vec_map (`dict`):
            文章ID到文章向量的映射。
        article_index (`AnnoyIndex`):
            用于快速查找相似文章的索引。
        user_item_dict (`dict`):
            用户ID到其交互文章ID的字典。
        worker_id (`int`):
            工作线程的标识符，用于保存临时结果。

    Returns:
        None: 该函数将结果保存为 pickle 文件。
    """
    # 用于存储每个用户的推荐结果
    data_list = []

    # 遍历查询数据中的每个用户ID和目标文章ID
    for user_id, item_id in tqdm(df_query.values):
        rank = defaultdict(int)  # 存储相似文章及其得分

        # 获取用户已互动的文章ID
        interacted_items = user_item_dict[user_id]
        interacted_items = interacted_items[-1:]  # 只取最后一项

        # 对于用户互动的每一篇文章，计算推荐
        for item in interacted_items:
            article_vec = article_vec_map[item]  # 获取文章向量

            # 在索引中查找与该文章相似的文章
            item_ids, distances = article_index.get_nns_by_vector(
                article_vec, 100, include_distances=True)
            # 计算相似度得分
            sim_scores = [2 - distance for distance in distances]

            # 更新推荐文章的得分
            for relate_item, wij in zip(item_ids, sim_scores):
                if relate_item not in interacted_items:  # 确保不推荐已互动的文章
                    rank[relate_item] += wij  # 累加得分

        # 根据得分排序，选出前50个推荐文章
        sim_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)[:50]
        item_ids = [item[0] for item in sim_items]  # 推荐文章ID
        item_sim_scores = [item[1] for item in sim_items]  # 推荐得分

        # 创建临时DataFrame来存储推荐结果
        df_temp = pd.DataFrame()
        df_temp['article_id'] = item_ids
        df_temp['sim_score'] = item_sim_scores
        df_temp['user_id'] = user_id

        # 设置标签，标记目标文章
        if item_id == -1:
            df_temp['label'] = np.nan  # 如果目标文章ID为 -1，标签为NaN
        else:
            df_temp['label'] = 0  # 默认标签为0
            # 标记目标文章为1
            df_temp.loc[df_temp['article_id'] == item_id, 'label'] = 1

        # 重排列的顺序并转换数据类型
        df_temp = df_temp[['user_id', 'article_id', 'sim_score', 'label']]
        df_temp['user_id'] = df_temp['user_id'].astype('int')
        df_temp['article_id'] = df_temp['article_id'].astype('int')

        # 将临时结果添加到数据列表中
        data_list.append(df_temp)

    # 合并所有用户的推荐结果
    df_data = pd.concat(data_list, sort=False)

    # 创建临时保存目录并将结果保存为pickle文件
    os.makedirs('../user_data/tmp/w2v', exist_ok=True)
    df_data.to_pickle('../user_data/tmp/w2v/{}.pkl'.format(worker_id))


if __name__ == '__main__':
    # 根据运行模式选择数据源和保存路径
    if mode == 'valid':
        # 加载离线验证数据
        df_click = pd.read_pickle('../user_data/data/offline/click.pkl')
        df_query = pd.read_pickle('../user_data/data/offline/query.pkl')

        os.makedirs('../user_data/data/offline', exist_ok=True)
        os.makedirs('../user_data/model/offline', exist_ok=True)

        # 定义 Word2Vec 文件和模型路径
        w2v_file = '../user_data/data/offline/article_w2v.pkl'
        model_path = '../user_data/model/offline'
    else:
        df_click = pd.read_pickle('../user_data/data/online/click.pkl')
        df_query = pd.read_pickle('../user_data/data/online/query.pkl')

        os.makedirs('../user_data/data/online', exist_ok=True)
        os.makedirs('../user_data/model/online', exist_ok=True)

        w2v_file = '../user_data/data/online/article_w2v.pkl'
        model_path = '../user_data/model/online'

    log.debug(f'df_click shape: {df_click.shape}')
    log.debug(f'{df_click.head()}')

    # 训练 Word2Vec 模型并获取文章向量映射
    article_vec_map = word2vec(df_click, 'user_id', 'click_article_id',
                               model_path)

    # 保存训练好的 Word2Vec 模型
    with open(w2v_file, 'wb') as f:
        pickle.dump(article_vec_map, f)

    # 将 embedding 建立索引
    article_index = AnnoyIndex(256, 'angular')
    article_index.set_seed(2020)

    # 将每个文章ID及其向量添加到索引中
    for article_id, emb in tqdm(article_vec_map.items()):
        article_index.add_item(article_id, emb)

    # 构建索引，100是树的数量
    article_index.build(100)

    # 创建用户-物品字典
    user_item_ = df_click.groupby('user_id')['click_article_id'].agg(
        lambda x: list(x)).reset_index()
    user_item_dict = dict(
        zip(user_item_['user_id'], user_item_['click_article_id']))

    # 召回过程
    n_split = max_threads  # 设置并行线程数
    all_users = df_query['user_id'].unique()  # 获取所有用户ID
    shuffle(all_users)  # 打乱用户ID顺序
    total = len(all_users)
    n_len = total // n_split  # 每个线程处理的用户数量

    # 清空临时文件夹
    for path, _, file_list in os.walk('../tmp/w2v'):
        for file_name in file_list:
            os.remove(os.path.join(path, file_name))

    # 运行用户分批处理
    for i in range(0, total, n_len):
        part_users = all_users[i:i + n_len]  # 读取当前批次用户
        df_temp = df_query[df_query['user_id'].isin(part_users)]  # 筛选查询数据
        # 召回排序结果
        recall(df_temp, article_vec_map, article_index, user_item_dict, i)

    # 等待所有任务完成
    multitasking.wait_for_tasks()
    log.info('合并任务')

    # 合并所有临时结果
    df_data = pd.DataFrame()
    for path, _, file_list in os.walk('../user_data/tmp/w2v'):
        for file_name in file_list:
            df_temp = pd.read_pickle(os.path.join(path, file_name))
            df_data = df_data.append(df_temp)

    # 对合并后的数据进行排序
    df_data = df_data.sort_values(['user_id', 'sim_score'],
                                  ascending=[True,
                                             False]).reset_index(drop=True)
    log.debug(f'df_data.head: {df_data.head()}')

    # 计算召回指标
    if mode == 'valid':
        log.info(f'计算召回指标')
        total = df_query[df_query['click_article_id'] != -1].user_id.nunique()
        # 计算不同阈值下的评估指标
        hitrate_5, mrr_5, hitrate_10, mrr_10, hitrate_20, mrr_20, hitrate_40, mrr_40, hitrate_50, mrr_50 = evaluate(
            df_data[df_data['label'].notnull()], total)

        log.debug(
            f'w2v: {hitrate_5}, {mrr_5}, {hitrate_10}, {mrr_10}, {hitrate_20}, {mrr_20}, {hitrate_40}, {mrr_40}, {hitrate_50}, {mrr_50}'
        )

    # 保存召回结果
    if mode == 'valid':
        df_data.to_pickle('../user_data/data/offline/recall_w2v.pkl')
    else:
        df_data.to_pickle('../user_data/data/online/recall_w2v.pkl')
