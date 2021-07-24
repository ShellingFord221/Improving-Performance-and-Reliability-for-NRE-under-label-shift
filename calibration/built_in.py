import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random



device = torch.device("cuda")


def apply_dropout(m):
	if type(m) == nn.Dropout:
		m.train()

def calcEntropy(batch_scores):
	# input: B * L
	# output: B
	batch_probs = nn.functional.softmax(batch_scores)
	return torch.sum(batch_probs * torch.log(batch_probs), dim=1).neg()

def calcEntropy2(batch_probs):
	return torch.sum(batch_probs * torch.log(batch_probs), dim=1).neg()

def calcVariance(batch_probs, sample_num):
	# input: [alldata_num, sample_num, class_num]
	# output: alldata_num
	SoftmaxVariance = []
	for p in batch_probs:
		p = np.array(p.tolist())

		# BALD的计算 G_X-F_X，参考https://github.com/Riashat/Active-Learning-Bayesian-Convolutional-Neural-Networks/blob/master/ConvNets/FINAL_Averaged_Experiments/Final_Experiments_Run/Softmax_Bald_Q10_N1000.py
		Entropy_Compute = - np.multiply(p, np.log2(p))
		# 按类求和(c)
		Entropy_Per_Softmax = np.sum(Entropy_Compute, axis=1)
		# 按sample求和(t)
		All_Entropy_Softmax = np.sum(Entropy_Per_Softmax, axis=0)
		F_X = np.divide(All_Entropy_Softmax, sample_num)

		# axis=0表示按列求和
		score_All = np.sum(p, axis=0)
		Avg_Pi = np.divide(score_All, sample_num)
		Entropy_Avg_Pi = - np.multiply(Avg_Pi, np.log2(Avg_Pi))
		G_X = np.sum(Entropy_Avg_Pi, axis=0)

		U_X = G_X - F_X
		SoftmaxVariance.append(U_X)

	# print(len(SoftmaxVariance))   # 1866
	return SoftmaxVariance

def calcInd(batch_probs):
	# input: B * L
	# output: B
	_, ind = torch.max(batch_probs, 1)
	return ind



def TuneMaxThres(model, test_dset, noneInd=0, ratio=0.2, cvnum=100):
    '''
    Tune threshold on test set
    '''
    model.eval()

    labels = []
    scores = []

    for batch in test_dset:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        # b_w = batch[5].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            loss, logits = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos)

        scores.append(logits)
        labels.append(b_labels)


    # start tuning
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    f1score = 0.0
    recall = 0.0
    precision = 0.0

    pre_prob, pre_ind = torch.max(scores, 1)
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_prob[ind], labels[ind]] for ind in range(0, len(pre_ind))]
    for cvind in tqdm(range(cvnum)):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] > threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold

        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2] != noneInd:
                ofInterest += 1
            if ins[1] > bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2]:
                    corrected += 1
        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))

    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return precision, recall, f1score




def TuneEntropyThres(model, test_dset, noneInd=0, ratio=0.2, cvnum=100):
    '''
    Tune threshold on test set (clean dev)
    '''
    model.eval()

    labels = []
    scores = []

    for batch in test_dset:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)
        # b_w = batch[5].to(device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            loss, logits = model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels,
                                        e1_pos=b_e1_pos,
                                        e2_pos=b_e2_pos)

        scores.append(logits)
        labels.append(b_labels)

    # start tuning
    scores = torch.cat(scores, dim=0)
    labels = torch.cat(labels, dim=0)

    f1score = 0.0
    recall = 0.0
    precision = 0.0

    pre_ind = calcInd(scores)
    pre_entropy = calcEntropy(scores)
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], labels[ind]] for ind in range(0, len(pre_ind))]

    for cvind in tqdm(range(cvnum)):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] < threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2] != noneInd:
                ofInterest += 1
            if ins[1] < bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2]:
                    corrected += 1

        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))


    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return precision, recall, f1score




def TuneMaxThres2(model, test_dset, rel2id_num, sample_num, noneInd=0, ratio=0.2, cvnum=100):
    '''
    Tune threshold on test set
    '''
    model.apply(apply_dropout)

    labels = []
    scores = []

    for batch in test_dset:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)

        scores_s =  np.zeros((len(batch[2]), rel2id_num))

        for s in range(sample_num):
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                loss, logits = model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos)

            scores_b = nn.functional.softmax(logits)
            scores_s += np.array(scores_b.tolist())

        scores_s /= sample_num
        scores_s = list(scores_s)

        scores += scores_s
        labels.append(b_labels)


    # start tuning
    scores = torch.tensor(scores)
    labels = torch.cat(labels, dim=0).tolist()

    f1score = 0.0
    recall = 0.0
    precision = 0.0

    pre_prob, pre_ind = torch.max(scores, 1)
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_prob[ind], labels[ind]] for ind in range(0, len(pre_ind))]
    # tqdm是进度条，cvnum是调整多少次
    # sample100次clean dev使得每个Model的结果相对稳定
    for cvind in tqdm(range(cvnum)):
        # 每次都随机从所有test data中选取20%作为clean dev
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] > threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold

        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2] != noneInd:
                ofInterest += 1
            if ins[1] > bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2]:
                    corrected += 1
        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))

    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return precision, recall, f1score



def TuneEntropyThres2(model, test_dset, rel2id_num, sample_num, noneInd=0, ratio=0.2, cvnum=100):
    '''
    Tune threshold on test set (clean dev)
    '''
    model.apply(apply_dropout)

    labels = []
    scores = []

    for batch in test_dset:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)

        scores_s = np.zeros((len(batch[2]), rel2id_num))

        for s in range(sample_num):
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                loss, logits = model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos)

            scores_b = nn.functional.softmax(logits)
            scores_s += np.array(scores_b.tolist())

        scores_s /= sample_num
        scores_s = list(scores_s)

        scores += scores_s
        labels.append(b_labels)

    # start tuning
    scores = torch.tensor(scores)
    labels = torch.cat(labels, dim=0).tolist()

    f1score = 0.0
    recall = 0.0
    precision = 0.0

    pre_ind = calcInd(scores)
    pre_entropy = calcEntropy2(scores)
    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_entropy[ind], labels[ind]] for ind in range(0, len(pre_ind))]

    for cvind in tqdm(range(cvnum)):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        # clean dev上找best threshold，然后在test data上测
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] < threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2] != noneInd:
                ofInterest += 1
            if ins[1] < bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2]:
                    corrected += 1

        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))


    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return precision, recall, f1score



def TuneVarianceThres(model, test_dset, rel2id_num, sample_num, noneInd=0, ratio=0.2, cvnum=100):
    '''
    Tune threshold on test set (clean dev)
    '''
    model.apply(apply_dropout)

    labels = []
    scores_sam = []
    scores_ave = []

    for batch in test_dset:
        # Unpack this training batch from our dataloader.
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)

        scores_l = []
        scores_s = np.zeros((len(batch[2]), rel2id_num))

        for s in range(sample_num):
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                loss, logits = model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,
                                            labels=b_labels,
                                            e1_pos=b_e1_pos,
                                            e2_pos=b_e2_pos)

            scores_b = nn.functional.softmax(logits)
            scores_s += np.array(scores_b.tolist())
            scores_l.append(scores_b.tolist())

        scores_s /= sample_num
        scores_s = list(scores_s)

        scores_ave += scores_s                      # 平均值 [alldata_num, class_num]
        scores_sam.append(torch.tensor(scores_l))
        labels.append(b_labels)


    # start tuning
    scores_sam = torch.cat(scores_sam, dim=1)   # 所有值 [sample_num, alldata_num, class_num]
    # print(scores_sam.size())
    scores_sam = scores_sam.permute(1, 0, 2)
    # print(scores_sam.size())                    # 所有值 [alldata_num, sample_num, class_num]
    scores_ave = torch.tensor(scores_ave)       # 平均值 [alldata_num, class_num]
    # print(scores_ave.size())
    labels = torch.cat(labels, dim=0).tolist()

    f1score = 0.0
    recall = 0.0
    precision = 0.0

    # print(labels[:20])
    pre_ind = calcInd(scores_ave)
    # print(pre_ind[:20])
    pre_variance = calcVariance(scores_sam, sample_num)
    # print(pre_variance[:20])

    valSize = int(np.floor(ratio * len(pre_ind)))
    data = [[pre_ind[ind], pre_variance[ind], labels[ind]] for ind in range(0, len(pre_ind))]

    for cvind in tqdm(range(cvnum)):
        random.shuffle(data)
        val = data[0:valSize]
        eva = data[valSize:]

        # find best threshold
        max_ent = max(val, key=lambda t: t[1])[1]
        min_ent = min(val, key=lambda t: t[1])[1]
        stepSize = (max_ent - min_ent) / 100
        thresholdList = [min_ent + ind * stepSize for ind in range(0, 100)]
        ofInterest = 0
        for ins in val:
            if ins[2] != noneInd:
                ofInterest += 1
        bestThreshold = float('nan')
        bestF1 = float('-inf')
        for threshold in thresholdList:
            corrected = 0
            predicted = 0
            for ins in val:
                if ins[1] < threshold and ins[0] != noneInd:
                    predicted += 1
                    if ins[0] == ins[2]:
                        corrected += 1
            curF1 = 2.0 * corrected / (ofInterest + predicted)
            if curF1 > bestF1:
                bestF1 = curF1
                bestThreshold = threshold
        # print(bestF1)
        # print(bestThreshold)
        ofInterest = 0
        corrected = 0
        predicted = 0
        for ins in eva:
            if ins[2] != noneInd:
                ofInterest += 1
            if ins[1] < bestThreshold and ins[0] != noneInd:
                predicted += 1
                if ins[0] == ins[2]:
                    corrected += 1

        f1score += (2.0 * corrected / (ofInterest + predicted))
        recall += (1.0 * corrected / ofInterest)
        precision += (1.0 * corrected / (predicted + 0.00001))


    f1score /= cvnum
    recall /= cvnum
    precision /= cvnum

    return precision, recall, f1score
