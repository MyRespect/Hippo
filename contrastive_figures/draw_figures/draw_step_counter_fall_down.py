import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'arial',
        'weight'   : 'bold',
        'size'   : 20}

matplotlib.rc('font', **font)

plt.rcParams["figure.figsize"] = (12, 4)

fig, axes = plt.subplots(1, 2)

scores=[]
ind = np.arange(3)

l200_step_sam = (np.array([206, 218, 236])-200)/2
l600_step_sam = (np.array([602, 647, 668])-600)/6
l1000_step_sam = (np.array([1004, 1021, 1192])-1000)/10

l200_step_zuk = (np.array([201, 209, 225])-200)/2
l600_step_zuk = (np.array([605, 617, 634])-600)/6
l1000_step_zuk = (np.array([1008, 1026, 1137])-1000)/10

width = 0.18

axes[0].yaxis.grid(zorder=0)
axes[0].bar(ind-0.02, [l200_step_sam[0], l600_step_sam[0], l1000_step_sam[0]], width, label='Raw',  bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0].bar(ind+width, [l200_step_sam[1], l600_step_sam[1], l1000_step_sam[1]], width, label='Gran. 1',  bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0].bar(ind+width*2+0.02, [l200_step_sam[2], l600_step_sam[2], l1000_step_sam[2]], width, label='Gran. 2', bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[0].set_xlabel("Steps", fontdict=font)
axes[0].set_ylim(0, 20)
axes[0].set_xticks(ind + width, ('200', '600',  '1000'))

axes[0].legend(loc='upper right')
axes[0].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[0].set_ylabel('Error Rate (%)', size = 20, weight = 'bold')
axes[0].set_title('Samsung S9', fontdict=font)


axes[1].yaxis.grid(zorder=0)
axes[1].bar(ind-0.02, [l200_step_zuk[0], l600_step_zuk[0], l1000_step_zuk[0]], width, label='Raw', bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#026E81', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1].bar(ind+width, [l200_step_zuk[1], l600_step_zuk[1], l1000_step_zuk[1]], width, label='Gran. 1', bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#00ABBD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1].bar(ind+width*2+0.02, [l200_step_zuk[2], l600_step_zuk[2], l1000_step_zuk[2]], width, label='Gran. 2', bottom=0, align='center', alpha=1, capsize=3, zorder=3, color = '#0099DD', error_kw=dict(ecolor='grey', lw=1, capsize=2, capthick=1))
axes[1].set_xlabel("Steps", fontdict=font)
axes[1].set_ylim(0, 20)
axes[1].set_xticks(ind + width, ('200', '600',  '1000'))

axes[1].legend(loc='upper right')
axes[1].yaxis.grid(linestyle='--')
# ax.set_xlabel('Client Number', size = 10, weight = 'bold')
axes[1].set_ylabel('Error Rate (%)', size = 20, weight = 'bold')
axes[1].set_title('ZUK Z2', fontdict=font)

plt.subplots_adjust(wspace=0.2, hspace=0.2)

fig = plt.gcf()



plt.tight_layout()
plt.show()
fig.savefig('step_counter.pdf', dpi=300, transparent=True, bbox_inches='tight')
