import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'arial',
        'weight'   : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

plt.rcParams["figure.figsize"] = (12.5,8)

fig, axes = plt.subplots(2, 2)

scores=[]
ind = np.arange(3) 


ae_har = [[74.4, 78.4, 76.2], [45.4, 49.6, 51.2], [34.4, 43.8, 45.1]]
gan_har =[[68.3, 54.6, 71.1], [61.6, 73.7, 53.5], [66, 71.1, 69.6]]
dm_har = [[72.6, 70.1, 68.7], [57.4, 61.2, 64.4], [58.7, 55.2, 60.6]]

ae_ms = [[68.6, 70.3, 69.1], [53.4, 52.9, 50.5], [42.1, 42.7, 43.8]]
gan_ms =[[60.4, 55.3, 64.6], [50.5, 60.4, 49.8], [52.4, 64.2, 60.4]]
dm_ms = [[84.1, 82.9, 81.7], [66.7, 65.8, 66.7], [55.4, 53.8, 54.7]]

ae_uci = [[66.8, 63.9, 61.2], [56.7, 58.6, 54], [46.6, 47.8, 40.9]]
gan_uci = [[60.6, 57.1, 63.5], [51.7, 55.8, 59.4], [51.7, 54.2, 62.6]]
dm_uci = [[82.2, 83.5, 79.7], [59.3, 61.9, 58.1], [54.7, 56.2, 54.7]]

ae_day = [[62.7, 59.5, 53.7], [34.8, 38.6, 34.8], [23.7, 28.2, 20.3]]
gan_day = [[65.6, 61.8, 63.4], [37.4, 31.7, 39.4], [38.4, 40.5, 37.2]]
dm_day = [[75.8, 76.1, 78.7], [67.9, 64.8, 61.8], [48.4, 46.5, 48.4]]

width = 0.18

axes[0][0].yaxis.grid(zorder=0)
axes[0][0].bar(ind-0.02, [np.mean(ae_har[0]), np.mean(gan_har[0]), np.mean(dm_har[0])], width, label='Granu. 1',  yerr=[np.std(ae_har[0]), np.std(gan_har[0]), np.std(dm_har[2])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0][0].bar(ind+width, [np.mean(ae_har[1]), np.mean(gan_har[1]), np.mean(dm_har[1])], width, label='Granu. 2',  yerr=[np.std(ae_har[1]), np.std(gan_har[1]), np.std(dm_har[1])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0][0].bar(ind+width*2+0.02, [np.mean(ae_har[2]), np.mean(gan_har[2]), np.mean(dm_har[2])], width, label='Granu. 3', yerr=[np.std(ae_har[2]), np.std(gan_har[2]), np.std(dm_har[2])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))

axes[0][0].set_xticks(ind + width, ('AE', 'GAN',  'Hippo'))

axes[0][0].legend(loc='lower right')
axes[0][0].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[0][0].set_ylabel('Accuracy (%)', size = 20, weight = 'bold')
axes[0][0].set_title('HARBox', fontdict=font)


axes[0][1].yaxis.grid(zorder=0)
axes[0][1].bar(ind-0.02, [np.mean(ae_ms[0]), np.mean(gan_ms[0]), np.mean(dm_ms[0])], width, label='Granu. 1',  yerr=[np.std(ae_ms[0]), np.std(gan_ms[0]), np.std(dm_ms[0])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0][1].bar(ind+width, [np.mean(ae_ms[1]), np.mean(gan_ms[1]), np.mean(dm_ms[1])], width, label='Granu. 2',  yerr=[np.std(ae_ms[1]), np.std(gan_ms[1]), np.std(dm_ms[1])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0][1].bar(ind+width*2+0.02, [np.mean(ae_ms[2]), np.mean(gan_ms[2]), np.mean(dm_ms[2])], width, label='Granu. 3', yerr=[np.std(ae_ms[2]), np.std(gan_ms[2]), np.std(dm_ms[2])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))

axes[0][1].set_xticks(ind + width, ('AE', 'GAN',  'Hippo'))

axes[0][1].legend(loc='lower right')
axes[0][1].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[0][1].set_ylabel('Accuracy (%)', size = 20, weight = 'bold')
axes[0][1].set_title('MotionSense', fontdict=font)


axes[1][0].yaxis.grid(zorder=0)
axes[1][0].bar(ind-0.02, [np.mean(ae_uci[0]), np.mean(gan_uci[0]), np.mean(dm_uci[0])], width, label='Granu. 1',  yerr=[np.std(ae_uci[0]), np.std(gan_uci[0]), np.std(dm_uci[0])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1][0].bar(ind+width, [np.mean(ae_uci[1]), np.mean(gan_uci[1]), np.mean(dm_uci[1])], width, label='Granu. 2',  yerr=[np.std(ae_uci[1]), np.std(gan_uci[1]), np.std(dm_uci[1])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1][0].bar(ind+width*2+0.02, [np.mean(ae_uci[2]), np.mean(gan_uci[2]), np.mean(dm_uci[2])], width, label='Granu. 3', yerr=[np.std(ae_uci[2]), np.std(gan_uci[2]), np.std(dm_uci[2])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))

axes[1][0].set_xticks(ind + width, ('AE', 'GAN',  'Hippo'))

axes[1][0].legend(loc='lower right')
axes[1][0].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[1][0].set_ylabel('Accuracy (%)', size = 20, weight = 'bold')
axes[1][0].set_title('UCIHAR', fontdict=font)


axes[1][1].yaxis.grid(zorder=0)
axes[1][1].bar(ind-0.02, [np.mean(ae_day[0]), np.mean(gan_day[0]), np.mean(dm_day[0])], width, label='Granu. 1',  yerr=[np.std(ae_day[0]), np.std(gan_day[0]), np.std(dm_day[0])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1][1].bar(ind+width, [np.mean(ae_day[1]), np.mean(gan_day[1]), np.mean(dm_day[1])], width, label='Granu. 2',  yerr=[np.std(ae_day[1]), np.std(gan_day[1]), np.std(dm_day[1])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1][1].bar(ind+width*2+0.02, [np.mean(ae_day[2]), np.mean(gan_day[2]), np.mean(dm_day[2])], width, label='Granu. 3', yerr=[np.std(ae_day[2]), np.std(gan_day[2]), np.std(dm_day[2])], bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))

axes[1][1].set_xticks(ind + width, ('AE', 'GAN',  'Hippo'))

axes[1][1].legend(loc='lower right')
axes[1][1].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[1][1].set_ylabel('Accuracy (%)', size = 20, weight = 'bold')
axes[1][1].set_title('DailySports', fontdict=font)

plt.subplots_adjust(wspace=0.2, hspace=0.2)

fig = plt.gcf()



plt.tight_layout()
plt.show()
fig.savefig('gen_comparison.pdf', dpi=300, transparent=True, bbox_inches='tight')
