from trajectory.get import get
import numpy as np
import pandas as pd
from directories import home
from tqdm import tqdm
import matplotlib.pyplot as plt


df_ant_HIT = pd.read_excel(home + '\\lists_of_experiments\\df_ant_HIT.xlsx')

shape = 'T'
# go through sizes 'M' and 'XL' and plot the traj.angle in a plot for each size
fig, axs = plt.subplots(5, 2, figsize=(10,20))
for i, size in enumerate(['M', 'XL']):
    df = df_ant_HIT[(df_ant_HIT['size'] == size) & (df_ant_HIT['shape'] == shape)]
    for iu, filename in tqdm(enumerate(df['filename'].head(5))):
        x = get(filename)
        x.smooth(1)

        angle = x.angle % (2 * np.pi)
        x_pos = x.position[:, 0]
        for ii in range(5):
            axs[ii][i].scatter(x_pos, angle, label=filename, s=2, color='lightgrey', alpha=0.5)
        axs[iu][i].scatter(x_pos, angle, label=filename, s=4, color='red')
        axs[iu][i].set_ylim(0, 2 * np.pi)
        axs[iu][i].set_xlabel('x position')
        axs[iu][0].set_xlim(14, 20)
plt.savefig(shape + '_angle_vs_x_position.png')
DEBUG = 1
