import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator



DATASET = 'KBP'

result_file_orig = './RC_result/bert_' + DATASET + '_orig.txt'
result_file_LA = './RC_result/bert_' + DATASET + '_LA.txt'
result_file_LA_BCTS = './RC_result/bert_' + DATASET + '_LA_calibrate.txt'
result_file_calibrator = './RC_result/bert_' + DATASET + '_calibrator.txt'


def get_cov_risk(result_file):

    result = open(result_file, 'r', encoding='UTF-8')

    max_prob_list = []
    right_or_wrong_list = []
    for line in result.readlines()[1:]:
        items = line.strip().split('\t')

        if result_file.endswith('calibrator.txt'):
            max_prob = float(items[0])
            right_or_wrong = int(items[1])

        else:
            # for KBP
            if DATASET == 'KBP':
                max_prob = float(items[9])
                right_or_wrong = int(items[10])

            # for NYT
            else:
                max_prob = float(items[27])
                right_or_wrong = int(items[28])

        max_prob_list.append(max_prob)
        right_or_wrong_list.append(right_or_wrong)

    result.close()

    # rc is a list of tuple
    rc = list(zip(max_prob_list, right_or_wrong_list))

    rc_sorted = sorted(rc, key=lambda x: x[0], reverse=True)

    coverage_list = []
    risk_list = []
    wrong = 0
    for i, pair in enumerate(rc_sorted):
        print(pair)
        coverage = float((i+1) / len(rc))
        wrong = wrong + pair[1]
        risk = float(wrong / len(rc))

        coverage_list.append(coverage)
        risk_list.append(risk)

    return coverage_list, risk_list


cov_orig, risk_orig = get_cov_risk(result_file_orig)
cov_LA, risk_LA = get_cov_risk(result_file_LA)
cov_LA_BCTS, risk_LA_BCTS = get_cov_risk(result_file_LA_BCTS)
cov_calibrator, risk_calibrator = get_cov_risk(result_file_calibrator)


aurc_orig = auc(x=np.array(cov_orig), y=np.array(risk_orig))
aurc_LA = auc(x=np.array(cov_LA), y=np.array(risk_LA))
aurc_LA_BCTS = auc(x=np.array(cov_LA_BCTS), y=np.array(risk_LA_BCTS))
aurc_calibrator = auc(x=np.array(cov_calibrator), y=np.array(risk_calibrator))


print('Orig AURC is: ', aurc_orig)
print('LA AURC is: ', aurc_LA)
print('LA_VS AURC is: ', aurc_LA_BCTS)
print('calibrator AURC is: ', aurc_calibrator)


# file_name = 'result/' + args.model_name + '_pr.txt'
# pr_file = open(file_name, 'w', encoding='UTF-8')
# for p, r in zip(precisions, recalls):
#     pr_file.write(str(p) + '\t' + str(r) + '\n')
# pr_file.close()
# print('pr file written.')


plt.clf()


plt.plot(np.array(cov_orig), np.array(risk_orig), "cornflowerblue", marker="d", markersize=8, markevery=400, lw=3, label='Orig.')
# y2 = np.zeros(len(cov_orig))
# plt.fill_between(np.array(cov_orig), np.array(risk_orig), y2, #上限，下限
#         facecolor='deepskyblue', #填充颜色
#         edgecolor='black', #边界颜色
#         alpha=0.3) #透明度


plt.plot(np.array(cov_LA), np.array(risk_LA), "tab:orange", marker="o", markersize=8, markevery=400, lw=3, label='Orig. + LA')
# y2 = np.zeros(len(cov_LA))
# plt.fill_between(np.array(cov_LA), np.array(risk_LA), y2, #上限，下限
#         facecolor='tomato', #填充颜色
#         edgecolor='black', #边界颜色
#         alpha=0.3) #透明度

if DATASET == 'KBP':
    plt.plot(np.array(cov_LA_BCTS), np.array(risk_LA_BCTS), "forestgreen", marker="s", markersize=8, markevery=200, lw=3, label='Orig. + LA + BCTS')
else:
    plt.plot(np.array(cov_LA_BCTS), np.array(risk_LA_BCTS), "forestgreen", marker="s", markersize=8, markevery=400, lw=3, label='Orig. + LA + VS')

plt.plot(np.array(cov_calibrator), np.array(risk_calibrator), "peru", marker="^", markersize=8, markevery=400, lw=3, label='Orig. + LA + Calibrator')


x_major_locator = MultipleLocator(0.2)
y_major_locator = MultipleLocator(0.05)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

plt.ylim([0.0, 0.2])
plt.xlim([0.0, 1.0])
plt.tick_params(axis='both', which='major', labelsize=16)

plt.legend(loc="upper left", fontsize=18)
if DATASET == 'KBP':
    plt.title("RC-curve on KBP", size=20)
else:
    plt.title("RC-curve on NYT", size=20)
plt.xlabel('Coverage', size=18)
plt.ylabel('Risk', size=18)
plt.grid(True, ls='--')
# plt.savefig('result/' + args.model_name + '_pr.jpg')
plt.show()


