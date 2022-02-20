import collections

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# sns.set(rc={'figure.figsize':(100, 100)})
sns.set_style('ticks', 
              rc={'axes.grid': True,
               'grid.color': '.8',
               'grid.linestyle': '-',
               'axes.spines.left': False,
               'axes.spines.right': False,
               'axes.spines.top': False,
              })
sns.set_context("paper", font_scale=1.4)

dir = '/home/lihepeng/Documents/Github/spinningup_tf2/out/Ant-v2/'
algs = ['ddpg', 'sac', 'td3', 'ppo', 'gpo']

fig = plt.figure()
for alg in algs:
    df = pd.read_csv(dir+'{}/exp-1/progress.txt'.format(alg), sep="\t")
    df = pd.DataFrame([df['AverageTestEpRet'].values[:1500]]).melt()
    sns.lineplot(data=df, x='variable', y='value')

# plt.xticks(np.arange(0,501,50), labels=['0']+['%.0fk'%i for i in np.arange(0,101,10)][1:])
# plt.yticks(np.arange(-1800,-500,200)[1:], labels=['%.0fk'%i for i in np.arange(-18,-5,2)[1:]])
plt.xlabel(r'TotalENVInteracts', fontsize='large')
plt.ylabel('Return', fontsize='large')
plt.legend(algs, loc=4)
plt.grid()
plt.tight_layout()
plt.show(block=True)

