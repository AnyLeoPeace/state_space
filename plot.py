import numpy as np
import matplotlib.pyplot as plt

'''Get results data'''
trans_scores = []
att_scores = []
ps = []
scores_name = ['Accuracy','F1-Score','Log-lilelihood']
iterations = 8


for proportion in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    for i in range(iterations):
        path = './results/proportion_' + str(proportion) + '_iteration_'+str(i) + '.npy'
        result = np.load(path, allow_pickle = True).item()
        trans = result['trans']
        att = result['att']
        trans_scores.append(trans)
        att_scores.append(att)
        p = result['proportion'][-1]
    ps.append(p)


trans_scores = np.array(trans_scores).reshape(-1, 8,3)
att_scores = np.array(att_scores).reshape(-1,8,3)

'''Plot'''

for i in range(3):
    f = plt.figure()
    plt.errorbar(ps, trans_scores.mean(axis=1)[:,i], yerr=trans_scores.std(axis=1)[:,i], fmt="o", label = 'Proposed model')
    plt.errorbar(ps, att_scores.mean(axis=1)[:,i], yerr=att_scores.std(axis=1)[:,i], fmt="o", label = 'Attentive model')
    plt.xlabel('Proportion')
    plt.ylabel(scores_name[i])
    plt.legend()
    f.savefig('./figure/all_'+scores_name[i],bbox_inches='tight')


for i in range(3):
    f = plt.figure()
    plt.plot(ps, trans_scores.max(axis=1)[:,i], 'r--', label = 'Proposed model')
    plt.plot(ps, att_scores.max(axis=1)[:,i], 'g--', label = 'Attentive model')
    plt.xlabel('Proportion')
    plt.ylabel(scores_name[i])
    plt.legend()
    f.savefig('./figure/max_'+scores_name[i],bbox_inches='tight')