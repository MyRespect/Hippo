import torch, matplotlib
import numpy as np 
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

font = {'family' : 'arial',
        'weight'   : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

plt.rcParams["figure.figsize"] = (7,4.5)

layer_num =[3]

train_saved_embed_label = torch.load("save_embed_label_layer"+str(layer_num)+".pt", map_location=torch.device('cpu'))
# train_saved_embed_label = torch.load("save_embed_label_raw.pt", map_location=torch.device('cpu'))

train_saved_embed_cpu = [embed.detach().numpy() for embed, label in train_saved_embed_label]
train_saved_embed = []

for item in train_saved_embed_cpu:
	for kk in item:
		train_saved_embed.append(kk)

train_saved_embed = train_saved_embed[:5000]

print(len(train_saved_embed))

tsne = TSNE(n_components=2, verbose=1, perplexity=12, n_iter=1000, init="pca")
reduced_data = tsne.fit_transform(train_saved_embed)

print("np.shape(reduced_data)", np.shape(reduced_data))

kmeans = KMeans(init="k-means++", n_clusters=5, n_init=10, random_state=0)
df_cluster = kmeans.fit_predict(reduced_data.astype('double'))

centroids = kmeans.cluster_centers_
cen_x = [i[0] for i in centroids] 
cen_y = [i[1] for i in centroids]


print(df_cluster)
# define and map colors
colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#C13584']
df_c = list(map({0:colors[0], 1:colors[1], 2:colors[2], 3:colors[3], 4:colors[4]}.get, df_cluster))

act_labels =['Walk', 'Hop', 'Call', 'Wave', 'Type']

#####PLOT#####
from matplotlib.lines import Line2D
fig, ax = plt.subplots()
# plot data
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=df_c, alpha = 0.6, s=8)
# create a list of legend elemntes
## markers / records
legend_elements = [Line2D([0], [0], marker='o', color='w', label=act_labels[i], 
               markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(colors)]

ax.set_xticks([])
ax.set_yticks([])
# plot legend
plt.legend(handles=legend_elements, loc='lower right', fontsize=18)
# title and labels
plt.title("Granu. "+str(layer_num)[1:2]+' Data Clustering', fontdict=font)
# plt.title('Raw Data Clustering', fontdict=font)
fig = plt.gcf()

plt.tight_layout()
plt.show()
fig.savefig('layer_'+str(layer_num)[1:2]+'cluster.pdf', dpi=300, transparent=True, bbox_inches='tight')
# fig.savefig('raw_cluster.pdf', dpi=300, transparent=True, bbox_inches='tight')













# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = 0.02  # point in the mesh [x_min, x_max]x[y_min, y_max].

# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# # Put the result into a color plot
# Z = Z.reshape(xx.shape)

# plt.clf()
# plt.imshow(
#     Z,
#     interpolation="nearest",
#     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#     cmap=plt.cm.tab20,
#     aspect="auto",
#     origin="lower",
# )

# plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(
#     centroids[:, 0],
#     centroids[:, 1],
#     marker="x",
#     s=300,
#     linewidths=3,
#     color="w",
#     zorder=10,
# )
# # plt.title(
# #     "K-means clustering on the digits dataset (PCA-reduced data)\n"
# #     "Centroids are marked with white cross"
# # )

# fig = plt.gcf()
# ax = fig.gca()
# circle2 = plt.Circle((10, 100), 10, color='r', fill=False, linewidth=2.5)
# ax.add_patch(circle2)

# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()