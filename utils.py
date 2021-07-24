import datetime
from collections import Counter
import json
import torch
import numpy as np
from torch.nn import functional as F


UNK_ID = 1
ner2id = {'<PAD>': 0, '<UNK>': 1, 'NATIONALITY': 2, 'SET': 3, 'ORDINAL': 4, 'ORGANIZATION': 5, 'MONEY': 6, 'PERCENT': 7, 'URL': 8, 'DURATION': 9, 'PERSON': 10, 'CITY': 11, 'CRIMINAL_CHARGE': 12, 'DATE': 13, 'TIME': 14, 'NUMBER': 15, 'STATE_OR_PROVINCE': 16, 'RELIGION': 17, 'MISC': 18, 'CAUSE_OF_DEATH': 19, 'LOCATION': 20, 'TITLE': 21, 'O': 22, 'COUNTRY': 23, 'IDEOLOGY': 24}
pos2id = {'<PAD>': 0, '<UNK>': 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}


def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else UNK_ID for t in tokens]
    return ids

# 'train.json'
def data_prepare(filename, DATASET, MAX_LENGTH, rel2id, tokenizer):
    train_set = json.load(open('./'+DATASET+'/' + filename, 'r'))
    sentence_train = []
    sentence_train_label = []

    for train_data in train_set:
        sentence = train_data['token']

        if train_data['relation'] not in rel2id.keys():
            continue
        else:
            label = train_data['relation']

        subj_start = train_data['subj_start']
        subj_end = train_data['subj_end']
        obj_start = train_data['obj_start']
        obj_end = train_data['obj_end']

        subj_type = train_data['subj_type']
        obj_type = train_data['obj_type']

        pos = train_data['stanford_pos']
        ner = train_data['stanford_ner']

        if subj_start < obj_start:
            sentence.insert(subj_start, '<e1>')
            sentence.insert(subj_end + 2, '</e1>')
            sentence.insert(obj_start + 2, '<e2>')
            sentence.insert(obj_end + 4, '</e2>')
        else:
            sentence.insert(obj_start, '<e2>')
            sentence.insert(obj_end + 2, '</e2>')
            sentence.insert(subj_start + 2, '<e1>')
            sentence.insert(subj_end + 4, '</e1>')
        sentence.insert(0, '[CLS]')
        sentence.append('[SEP]')

        sentence_train.append(' '.join(sentence))
        sentence_train_label.append(label)


    input_ids = []
    attention_masks = []
    e1_pos = []
    e2_pos = []
    labels = []
    discard = 0


    rel_cnt = {}
    log_prior = np.zeros(len(rel2id), dtype=np.float32)
    rel_distrib = np.zeros(len(rel2id), dtype=np.float32)

    for i, sent in enumerate(sentence_train):

        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # print(encoded_dict['input_ids'])
        # print(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0].tolist()))
        # print('e1', tokenizer.encode('e1'))
        # print('e2', tokenizer.encode('e2'))

        if 2487 not in encoded_dict['input_ids'][0].tolist() or 2475 not in encoded_dict['input_ids'][0].tolist():
            discard += 1
            # print('discard')
            continue

        else:
            e1_pos.append((encoded_dict['input_ids'] == 2487).nonzero()[0][1].item())
            e2_pos.append((encoded_dict['input_ids'] == 2475).nonzero()[0][1].item())

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

            rel = rel2id[sentence_train_label[i]]
            labels.append(rel)

            if sentence_train_label[i] not in rel_cnt:   # str
                rel_cnt[sentence_train_label[i]] = 0
            rel_cnt[sentence_train_label[i]] += 1


    for rel in rel_cnt.keys():
        relid = rel2id[rel]
        rel_distrib[relid] = rel_cnt[rel]
        log_prior[relid] = np.log(rel_cnt[rel])
    max_log = np.max(log_prior)
    log_prior = log_prior - max_log  # 其实就是除以分母
    rel_distrib = rel_distrib / np.sum(rel_distrib)


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)

    print(filename + ' discard: ' + str(discard))

    return input_ids, attention_masks, labels, e1_pos, e2_pos, log_prior, rel_distrib



def data_prepare2(filename, DATASET, MAX_LENGTH, rel2id, tokenizer):
    train_set = json.load(open('./'+DATASET+'/' + filename, 'r'))
    sentence_train = []
    sentence_train_label = []
    sentence_type = []
    sentence_pos = []

    for train_data in train_set:
        sentence = train_data['token']

        if train_data['relation'] not in rel2id.keys():
            continue
        else:
            label = train_data['relation']

        subj_start = train_data['subj_start']
        subj_end = train_data['subj_end']
        obj_start = train_data['obj_start']
        obj_end = train_data['obj_end']

        subj_type = ner2id[train_data['subj_type']]   # id
        obj_type = ner2id[train_data['obj_type']]   # id

        pos = map_to_ids(train_data['stanford_pos'], pos2id)   # id_list
        ner = map_to_ids(train_data['stanford_ner'], ner2id)   # id_list

        pos_e1 = pos[subj_start]   # id
        pos_e2 = pos[obj_start]   # id


        if subj_start < obj_start:
            sentence.insert(subj_start, '<e1>')
            sentence.insert(subj_end + 2, '</e1>')
            sentence.insert(obj_start + 2, '<e2>')
            sentence.insert(obj_end + 4, '</e2>')
        else:
            sentence.insert(obj_start, '<e2>')
            sentence.insert(obj_end + 2, '</e2>')
            sentence.insert(subj_start + 2, '<e1>')
            sentence.insert(subj_end + 4, '</e1>')
        sentence.insert(0, '[CLS]')
        sentence.append('[SEP]')

        sentence_train.append(' '.join(sentence))
        sentence_train_label.append(label)
        sentence_type.append([subj_type, obj_type])
        sentence_pos.append([pos_e1, pos_e2])


    input_ids = []
    attention_masks = []
    e1_pos = []
    e2_pos = []
    labels = []
    discard = 0

    types = []
    poses = []

    rel_cnt = {}
    log_prior = np.zeros(len(rel2id), dtype=np.float32)
    rel_distrib = np.zeros(len(rel2id), dtype=np.float32)

    for i, sent in enumerate(sentence_train):

        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LENGTH,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # print(encoded_dict['input_ids'])
        # print(tokenizer.convert_ids_to_tokens(encoded_dict['input_ids'][0].tolist()))
        # print('e1', tokenizer.encode('e1'))
        # print('e2', tokenizer.encode('e2'))

        if 2487 not in encoded_dict['input_ids'][0].tolist() or 2475 not in encoded_dict['input_ids'][0].tolist():
            discard += 1
            # print('discard')
            continue

        else:
            e1_pos.append((encoded_dict['input_ids'] == 2487).nonzero()[0][1].item())
            e2_pos.append((encoded_dict['input_ids'] == 2475).nonzero()[0][1].item())

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

            rel = rel2id[sentence_train_label[i]]
            labels.append(rel)

            types.append(sentence_type[i])
            poses.append(sentence_pos[i])

            if sentence_train_label[i] not in rel_cnt:   # str
                rel_cnt[sentence_train_label[i]] = 0
            rel_cnt[sentence_train_label[i]] += 1


    for rel in rel_cnt.keys():
        relid = rel2id[rel]
        rel_distrib[relid] = rel_cnt[rel]
        log_prior[relid] = np.log(rel_cnt[rel])
    max_log = np.max(log_prior)
    log_prior = log_prior - max_log  # 其实就是除以分母
    rel_distrib = rel_distrib / np.sum(rel_distrib)


    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    e1_pos = torch.tensor(e1_pos)
    e2_pos = torch.tensor(e2_pos)

    types = torch.tensor(types)
    poses = torch.tensor(poses)

    print(filename + ' discard: ' + str(discard))

    return input_ids, attention_masks, labels, e1_pos, e2_pos, log_prior, rel_distrib, types, poses




def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))



def score(key, prediction, verbose=True, NO_RELATION=0):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == NO_RELATION and guess == NO_RELATION:
            pass
        elif gold == NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
        elif gold != NO_RELATION and guess == NO_RELATION:
            gold_by_relation[gold] += 1
        elif gold != NO_RELATION and guess != NO_RELATION:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print("SET NO_RELATION ID: ", NO_RELATION)
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def calculate_ece(logits, labels, n_bins=10):

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def calculate_ece2(confidences, accuracies, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = torch.tensor(confidences)
    accuracies = torch.tensor(accuracies)

    ece = torch.zeros(1, device=confidences.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].double().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin.double()).float()
    return ece.item()


def get_cov_risk(max, label, f):
	rc = list(zip(max, label))

	# filein = open(f, 'w', encoding='UTF-8')
	# for pair in rc:
	# 	filein.write(str(pair[0]) + '\t' + str(pair[1]) + '\n')
	# filein.close()

	rc_sorted = sorted(rc, key=lambda x: x[0], reverse=True)

	coverage_list = []
	risk_list = []
	wrong = 0

	for i, pair in enumerate(rc_sorted):

		coverage = float((i+1) / len(rc))
		wrong = wrong + pair[1]
		risk = float(wrong / len(rc))

		coverage_list.append(coverage)
		risk_list.append(risk)

	return coverage_list, risk_list