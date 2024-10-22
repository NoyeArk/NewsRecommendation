import argparse
import pandas as pd
from tqdm import tqdm


def data_online(df_train_click, df_test_click):
    # 把测试集中的用户ID给提取出来
    test_users = df_test_click['user_id'].unique()
    test_query_list = []

    # 对于每个用户，都需要预测要点击的文章ID
    for user in tqdm(test_users):
        test_query_list.append([user, -1])

    # 将列表转化为dataframe类型的数据
    df_test_query = pd.DataFrame(test_query_list, columns=['user_id', 'click_article_id'])

    df_query = df_test_query
    df_click = pd.concat([df_train_click, df_test_query], sort=False).reset_index(drop=True)


def data_offline(df_train_click, df_test_click):
    pass


if __name__ == '__main__':
    df_train_click = pd.read_csv('../dataset/train_click_log.csv')
    df_test_click = pd.read_csv('../dataset/testA_click_log.csv')

    # 命令行参数
    parser = argparse.ArgumentParser(description="数据处理")
    parser.add_argument('--mode', default='valid')
    parser.add_argument('--logfile', default='test.log')

    args = parser.parse_args()

    if args.mode == 'train':
        data_online(df_train_click, df_test_click)
    else:
        data_offline(df_train_click, df_test_click)
