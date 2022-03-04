import collections

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

def moving_avg(nums, N):
    cumsum, moving_aves = [0], []
    for i, x in enumerate(nums, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    return moving_aves

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

env = 'Walker2d-v2'
dir = '/home/lihepeng/Documents/Github/spinningup_tf2/out/{}/'.format(env)
# algs = ['ddpg', 'sac', 'td3', 'ppo', 'gpo']
algs = ['sac', 'gpo']

fig = plt.figure()
for alg in algs:
    returns = []
    for seed in range(1,4):
        df = pd.read_csv(dir+'{}/exp-{}/progress.txt'.format(alg, seed), sep="\t")
        data = df['AverageTestEpRet'].values[:800]
        returns.append(moving_avg(data, 50))
    dfs = pd.DataFrame(np.array(returns)).melt()
    sns.lineplot(data=dfs, x='variable', y='value')

# plt.xticks(np.arange(0,501,50), labels=['0']+['%.0fk'%i for i in np.arange(0,101,10)][1:])
# plt.yticks(np.arange(-1800,-500,200)[1:], labels=['%.0fk'%i for i in np.arange(-18,-5,2)[1:]])
plt.xlabel(r'TotalENVInteracts', fontsize='large')
plt.ylabel('Return', fontsize='large')
plt.legend(algs, loc=4)
plt.grid()
plt.tight_layout()
plt.show(block=True)

