import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

folder_path = '../output/tanh_2d_global_0/'
# folder_path = '/Users/raphaelholca/Mountpoint/hebbianRL/output/gabor_xplr-7/'


global_optim = True if os.path.exists(folder_path + 'visited_params_global') else False

if not global_optim:
	visited_params = pickle.load(open(folder_path + 'visited_params', 'r'))
	visited_params = [visited_params]
else:
	visited_params = pickle.load(open(folder_path + 'visited_params_global', 'r'))

NUM_COLORS=len(visited_params)
c_list = [plt.get_cmap('Set1')(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] #alternate cmap: 'winter'

""" 2D plot exploaration path """
n_params = np.size(visited_params[0],1)-1
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

for i_basin, basin in enumerate(visited_params):
	alpha = np.clip(basin[:,3], 0.5, 1.0) if np.size(basin,1)>3 else 1.
	ax.plot(basin[:,0], basin[:,1], marker='.', c=c_list[i_basin], alpha=alpha)

ax.set_xlabel('param 0 value')
ax.set_ylabel('param 1 value')

plt.savefig(folder_path + 'params_path_2D.png')
# plt.show(block=False)
plt.close(fig)

""" 1D plot exploaration path """
fig, ax = plt.subplots(n_params,1)
fig.patch.set_facecolor('white')

for param in range(n_params):
	len_basin_counter = 0
	for i_basin, basin in enumerate(visited_params):
		alpha = np.clip(basin[:,3], 0.5, 1.0) if np.size(basin,1)>3 else 1.
		ax[param].plot(np.arange(len_basin_counter, len_basin_counter+np.size(basin,0)), basin[:,param], marker='.', c=c_list[i_basin], alpha=alpha)
		ax[param].set_ylabel('param ' + str(param) +' value')
		len_basin_counter+=np.size(basin,0)

ax[-1].set_xlabel('iterations')

plt.savefig(folder_path + 'params_1D.png')
# plt.show(block=False)
plt.close(fig)

""" plot performance """
fig, ax = plt.subplots()
fig.patch.set_facecolor('white')

len_basin_counter = 0
for i_basin, basin in enumerate(visited_params):
	alpha = np.clip(basin[:,3], 0.5, 1.0) if np.size(basin,1)>3 else 1.
	ax.plot(np.arange(len_basin_counter, len_basin_counter+np.size(basin,0)), basin[:,-1], marker='.', c=c_list[i_basin], alpha=alpha)
	len_basin_counter+=np.size(basin,0)

ax.set_xlabel('iterations')
ax.set_ylabel('error rate')

plt.savefig(folder_path + 'perf.png')
# plt.show(block=False)
plt.close(fig)















