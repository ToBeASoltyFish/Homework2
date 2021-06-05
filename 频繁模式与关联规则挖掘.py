#!/usr/bin/env python
# coding: utf-8

# #### 李坤 3220201059
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/winemag-data-130k-v2.csv', index_col=0)
df.head()

# 筛选出国家、葡萄品种、酒庄信息，构成事务集
transactions_1 = []
for index, row in df.iterrows():
    transactions_1 += [(row['country'], row['variety'], row['winery'])]
transactions_1[:20]

# 筛选出品酒师姓名、评分、葡萄品种、酒庄信息，构成事务集
transactions_2 = []
for index, row in df.iterrows():
    transactions_2 += [(row['taster_name'], row['points'], row['variety'], row['winery'])]
transactions_2[:20]


# ## 2.找出频繁模式


from collections import defaultdict
import itertools


def apriori(transactions, support=0.1, confidence=0.8, lift=1, minlen=2, maxlen=2):
    item_2_tranidxs = defaultdict(list)
    itemset_2_tranidxs = defaultdict(list)

    for tranidx, tran in enumerate(transactions):
        for item in tran:
            item_2_tranidxs[item].append(tranidx)
            itemset_2_tranidxs[frozenset([item])].append(tranidx)

    item_2_tranidxs = dict([(k, frozenset(v)) for k, v in item_2_tranidxs.items()])
    itemset_2_tranidxs = dict([
        (k, frozenset(v)) for k, v in itemset_2_tranidxs.items()])

    tran_count = float(len(transactions))

    valid_items = set(item
        for item, tranidxs in item_2_tranidxs.items()
            if (len(tranidxs) / tran_count >= support))

    pivot_itemsets = [frozenset([item]) for item in valid_items]
    freqsets = []

    if minlen == 1:
        freqsets.extend(pivot_itemsets)

    for i in range(maxlen - 1):
        new_itemset_size = i + 2
        new_itemsets = []

        for pivot_itemset in pivot_itemsets:
            pivot_tranidxs = itemset_2_tranidxs[pivot_itemset]
            for item, tranidxs in item_2_tranidxs.items():
                if item not in pivot_itemset:
                    common_tranidxs = pivot_tranidxs & tranidxs
                    if len(common_tranidxs) / tran_count >= support:
                        new_itemset = frozenset(pivot_itemset | set([item]))
                        if new_itemset not in itemset_2_tranidxs:
                            new_itemsets.append(new_itemset)
                            itemset_2_tranidxs[new_itemset] = common_tranidxs

        if new_itemset_size > minlen - 1:
            freqsets.extend(new_itemsets)

        pivot_itemsets = new_itemsets

    for freqset in freqsets:
        for item in freqset:
            rhs = frozenset([item])
            lhs = freqset - rhs
            support_rhs = len(itemset_2_tranidxs[rhs]) / tran_count
            if len(lhs) == 0:
                lift_rhs = float(1)
                if support_rhs >= support and support_rhs > confidence and lift_rhs > lift:
                    yield (lhs, rhs, support_rhs, support_rhs, lift_rhs)
            else:
                confidence_lhs_rhs = len(itemset_2_tranidxs[freqset])                     / float(len(itemset_2_tranidxs[lhs]))
                lift_lhs_rhs = confidence_lhs_rhs / support_rhs

                if confidence_lhs_rhs >= confidence and lift_lhs_rhs > lift:
                    support_lhs_rhs = len(itemset_2_tranidxs[freqset]) / tran_count
                    yield (lhs, rhs, support_lhs_rhs, confidence_lhs_rhs, lift_lhs_rhs)


# 设置频繁项集支持度、置信度、lift阈值：（support>0.03, confidence>0.1, lift>1）

#第一类：country-variety-winery中的频繁模式
rules_1 = apriori(transactions_1, support=0.03, confidence=0.1, lift=1)
rules1_sorted = sorted(rules_1, key=lambda x: (x[4], x[3], x[2]), reverse=True)

for r in rules1_sorted:
    print(r)

#第二类：taster-points-variety-winery中的频繁模式
rules_2 = apriori(transactions_2, support=0.03, confidence=0.1, lift=1)
rules2_sorted = sorted(rules_2, key=lambda x: (x[4], x[3], x[2]), reverse=True)

for r in rules2_sorted:
    print(r)

# ## 3.导出关联规则，计算其支持度和置信度

import csv 

with open('result1.csv', 'wt') as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule', 'sup', 'conf', 'lift'])
    for r in rules1_sorted:
        f_csv.writerow([f'{str(list(r[0])[0])} => {str(list(r[1])[0])}', r[2], r[3], r[4]])

pd.read_csv('result1.csv')

import csv 

with open('result2.csv', 'wt') as f:
    f_csv = csv.writer(f, delimiter=',')
    f_csv.writerow(['rule', 'sup', 'conf', 'lift'])
    for r in rules2_sorted:
        f_csv.writerow([f'{str(list(r[0])[0])} => {str(list(r[1])[0])}', r[2], r[3], r[4]])

pd.read_csv('result2.csv')


# 4.对规则进行评价，使用Lift、Kulc指标

res = []
for r in rules1_sorted:
    conf1 = r[3]
    for r2 in rules1_sorted:
        if r2[0] == r[1] and r2[1] == r[0]:
            conf2 = r2[3]
    kulc = (conf1 + conf2) / 2
    res.append(kulc)

res = []
for r in rules2_sorted:
    conf1 = r[3]
    for r2 in rules2_sorted:
        if r2[0] == r[1] and r2[1] == r[0]:
            conf2 = r2[3]
    kulc = (conf1 + conf2) / 2
    res.append(kulc)

# 5.对挖掘结果进行分析
pd.read_csv('result1.csv')

pd.read_csv('result2.csv')

df[df['variety'] == 'Bordeaux-style Red Blend'].sample(15)

df[df['variety'] == 'Cabernet Sauvignon'].sample(15)

# ## 6.可视化展示

df[df['variety'] == 'Bordeaux-style Red Blend']['country'].value_counts().plot(kind='bar')

df[df['country'] == 'US']['variety'].value_counts().head(20).plot(kind='bar')

df[df['taster_name'] == 'Roger Voss']['variety'].value_counts().head(20).plot(kind='bar')