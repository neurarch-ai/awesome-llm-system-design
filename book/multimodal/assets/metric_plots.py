import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
import matplotlib.patches as mp
OUT="/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/awesome-llm-system-design/book/multimodal/assets/"
plt.rcParams.update({'figure.dpi':130,'font.size':11,'axes.grid':False,'figure.autolayout':True})
BLUE,ORANGE,GREEN,RED,GRAY='#2563eb','#ea7317','#16a34a','#dc2626','#64748b'

# 1) IoU geometry: two boxes, intersection shaded, union outlined
fig,ax=plt.subplots(figsize=(5.2,3.6)); ax.set_aspect('equal'); ax.axis('off')
gt=mp.Rectangle((1,1),3.2,2.2,fill=False,edgecolor=GREEN,lw=2.5,label='ground truth')
pr=mp.Rectangle((2.3,1.7),3.2,2.2,fill=False,edgecolor=BLUE,lw=2.5,label='prediction')
ax.add_patch(gt); ax.add_patch(pr)
ix0,iy0,ix1,iy1=2.3,1.7,4.2,3.2
ax.add_patch(mp.Rectangle((ix0,iy0),ix1-ix0,iy1-iy0,facecolor=ORANGE,alpha=0.45,edgecolor='none'))
ax.text((ix0+ix1)/2,(iy0+iy1)/2,'∩',color='black',ha='center',va='center',fontsize=14)
ax.text(1.15,3.05,'ground truth',color=GREEN,fontsize=9)
ax.text(4.3,3.75,'prediction',color=BLUE,fontsize=9)
ax.text(3.25,0.35,r'IoU = area(overlap) / area(union);  match if IoU $\geq$ 0.5',ha='center',fontsize=10)
ax.set_xlim(0.5,6); ax.set_ylim(0,4.3); ax.set_title('IoU: how a detection or grounding box is scored')
fig.savefig(OUT+"fig-iou-computation.png"); plt.close(fig)

# 2) VQA soft voting: 10 human answers -> min(n/3,1)
fig,ax=plt.subplots(1,2,figsize=(8.6,3.4))
ans=['yellow','gold','tan','white']; votes=[6,3,1,0]
ax[0].bar(ans,votes,color=[GREEN,GREEN,GRAY,GRAY]); ax[0].set_ylabel('# of 10 humans')
ax[0].set_title('the 10 human answers for one question'); ax[0].axhline(3,ls='--',color=RED,lw=1)
ax[0].text(2.4,3.15,'3 = full credit',color=RED,fontsize=9)
n=np.arange(0,11); score=np.minimum(n/3,1)
ax[1].plot(n,score,marker='o',color=BLUE)
ax[1].set_xlabel('# humans who gave the predicted answer'); ax[1].set_ylabel('VQA score')
ax[1].set_title(r'score = min(n/3, 1)  (not exact match)'); ax[1].set_ylim(0,1.08); ax[1].grid(alpha=0.3)
fig.savefig(OUT+"fig-vqa-soft-voting.png"); plt.close(fig)

# 3) POPE: confusion of yes/no answers on present/absent objects
fig,ax=plt.subplots(figsize=(4.8,3.8))
M=np.array([[45,5],[18,32]])  # rows: object present/absent ; cols: model says yes/no
im=ax.imshow(M,cmap='Blues')
ax.set_xticks([0,1]); ax.set_xticklabels(['says "yes"','says "no"'])
ax.set_yticks([0,1]); ax.set_yticklabels(['object\npresent','object\nabsent'])
labels=[['true positive','miss'],['HALLUCINATION\n(false positive)','true negative']]
for i in range(2):
    for j in range(2):
        ax.text(j,i,f'{M[i,j]}\n{labels[i][j]}',ha='center',va='center',
                color='white' if M[i,j]>30 else 'black',fontsize=9)
ax.set_title('POPE: "Is there a {object}?" yes/no\nhallucination = yes on an absent object (low precision)')
fig.savefig(OUT+"fig-pope-confusion.png"); plt.close(fig)
print("wrote 3 multimodal metric figures")
