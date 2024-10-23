# 新闻推荐

## 1 比赛介绍

赛题以新闻APP中的新闻推荐为背景，要求选手根据用户历史浏览点击新闻文章的数据信息预测用户未来点击行为，即用户的最后一次点击的新闻文章。

测试集对最后一次点击行为进行了剔除。通过这道赛题来引导大家了解推荐系统中的一些业务背景，解决实际问题，帮助竞赛新人进行自我练习、自我提高。

---

## 2 数据集概述

赛题以预测用户未来点击新闻文章为任务，该数据来自某新闻APP平台的用户交互数据，包括30万用户，近300万次点击，共36万多篇不同的新闻文章，同时每篇新闻文章有对应的 embedding 向量表示。

为了保证比赛的公平性，将会从中抽取20万用户的点击日志数据作为训练集，5万用户的点击日志数据作为测试集A，5万用户的点击日志数据作为测试集B。

### 2.1 文件描述

| 文件名称                  | 描述                  |
|-----------------------|---------------------|
| `train_click_log.csv` | 训练集用户点击日志           |
| `testA_click_log.csv` | 测试集用户点击日志           |
| `articles.csv`        | 新闻文章信息数据表           |
| `articles_emb.csv`    | 新闻文章 embedding 向量表示 |
| `sample_submit.csv`   | 提交样例文件              |

### 2.2 字段描述

| 特征名称                    | 描述                        |
|-------------------------|---------------------------|
| user_id                 | 用户id                      |
| click_article_id        | 点击文章id                    |
| click_timestamp         | 点击时间戳                     |
| click_environment       | 点击环境                      |
| click_deviceGroup       | 点击设备组                     |
| click_os                | 点击操作系统                    |
| click_country           | 点击城市                      |
| click_region            | 点击地区                      |
| click_referrer_type     | 点击来源类型                    |
| article_id              | 文章id，与click_article_id相对应 |
| category_id             | 文章类型id                    |
| created_at_ts           | 文章创建时间戳                   |
| words_count             | 文章字数                      |
| emb_1,emb_2,...,emb_249 | 文章embedding向量表示           |

---

## 3 算法设计

参考[分享](https://tianchi.aliyun.com/forum/post/170754)