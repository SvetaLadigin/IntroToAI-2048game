import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = open('statistics.txt', 'r')
time= []
tile = []
agent = []
score = []
depth = []
avg_depth = []
avg_score = []
data = {}
for line in f:
    params = line.split()
    print(params)
    time.append(float(params[0]))
    agent.append(params[1])
    tile.append(float(params[2]))
    score.append(float(params[3]))
    depth.append(float(params[4]))
data['time'] = time
data['agent'] = agent
data['tile'] = tile
data['score'] = score
data['depth'] = depth
new_data = pd.DataFrame.from_dict(data)
minimax_data = new_data.loc[new_data['agent'] == 'minimax']
print(minimax_data)
avg_depth_minimax = minimax_data.groupby('time').depth.mean()
print(avg_depth_minimax)
ab_data = new_data.loc[new_data['agent'] == 'abmove']
print(ab_data)
avg_depth_ab = ab_data.groupby('time').depth.mean()
print(avg_depth_ab)
graph = avg_depth_minimax.plot(xlabel='time', ylabel='depth', label='minimax', style='g')
graph2 = avg_depth_ab.plot(xlabel='time', ylabel='depth',label='abmove', style='r')
plt.style.use('fivethirtyeight')
plt.legend()
plt.savefig('depth_ab_minimax.png')
plt.show()

avg_score_minimax = minimax_data.groupby('time').score.mean()
avg_score_ab = ab_data.groupby('time').score.mean()
graph3 = avg_score_minimax.plot(xlabel='time', ylabel='score', label='minimax', style='g')
graph4 = avg_score_ab.plot(xlabel='time', ylabel='score', label='abmove', style='r')
plt.legend()
plt.savefig('score_ab_minimax.png')
plt.show()
