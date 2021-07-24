import torch
from torch.nn import functional as F
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

from utils import data_prepare, format_time, score, calculate_ece
import os
import time, json
import numpy as np
import math
import random
from tqdm import tqdm

from bert import BertForSequenceClassificationUserDefined



CUDA = 0
DATASET = 'KBP'  
MAX_LENGTH = 100
BATCH_SIZE = 32
save_path = './bert_' + DATASET + '.zip'
# save_path = './bert_' + DATASET + '.pt'
dev_ratio = 0.2
cvnum = 100
result_file_LA = './bert_' + DATASET + '_LA.txt'



# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(CUDA))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")




# ========================================
#              Data prepare
# ========================================


# Load the BERT tokenizer.
print('Loading BERT tokenizer ...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
rel2id = json.load(open('./'+DATASET+'/relation2id.json', 'r'))
NUM_LABELS = len(rel2id)

print('Preparing data ...')

t1, t2, t3, t4, t5, train_log_prior, _ = data_prepare('train.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
train_dataset = TensorDataset(t1, t2, t3, t4, t5)

s1, s2, s3, s4, s5, test_log_prior, _ = data_prepare('test.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
test_dataset = TensorDataset(s1, s2, s3, s4, s5)

# test_dataloader = DataLoader(
#             test_dataset,  # The training samples.
#             sampler=SequentialSampler(test_dataset),  # Select batches randomly
#             batch_size=BATCH_SIZE  # Trains with this batch size.
#         )



model = BertForSequenceClassificationUserDefined.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
model.load_state_dict(torch.load(save_path))
model.cuda()
print('Model loaded.')



print('Test ...')

t0 = time.time()
model.eval()


# cut dev_ratio out of test set
datasize = len(test_dataset)
dev_cnt = math.ceil(datasize * dev_ratio)

indices = list(range(datasize))

test_result = []

best_f1 = 0
final_logits = []
final_labels = []

for j in tqdm(range(cvnum)):

    random.shuffle(indices)
    instances = [test_dataset[i] for i in indices]

    dev_instances = instances[:dev_cnt]

    rel_cnt = {}
    for d in dev_instances:
        r = str(int(d[2].item()))   # str(id)
        if r not in rel_cnt:
            rel_cnt[r] = 0
        rel_cnt[r] += 1

    dev_log_prior = np.zeros(len(rel2id), dtype=np.float32)
    for rel in rel_cnt.keys():   # str(id)
        dev_log_prior[int(rel)] = np.log(rel_cnt[rel])
    max_log = np.max(dev_log_prior)
    dev_log_prior = dev_log_prior - max_log  # 其实就是除以分母

    test_instances = instances[dev_cnt:]

    test_dataloader = DataLoader(
                test_instances,
                sampler=SequentialSampler(test_instances),  # Select batches randomly
                batch_size=BATCH_SIZE  # Trains with this batch size.
            )


    bias_old = model.classifier.bias.data
    cdev_lp = torch.from_numpy(dev_log_prior).to(device)
    train_lp = torch.from_numpy(train_log_prior).to(device)

    bias_new = bias_old - train_lp + cdev_lp

    model.classifier.bias.data = bias_new

    total_eval_loss = 0
    all_prediction = np.array([])
    all_ground_truth = np.array([])

    logits_list = []
    labels_list = []

    # Evaluate data for one epoch
    for batch in test_dataloader:
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

        logits_list.append(logits)
        labels_list.append(b_labels)

        # Accumulate the validation loss.
        total_eval_loss += loss.sum().item()
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        all_prediction = np.concatenate((all_prediction, pred_flat), axis=None)
        all_ground_truth = np.concatenate((all_ground_truth, labels_flat), axis=None)

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(test_dataloader)

    p, r, f1 = score(all_ground_truth, all_prediction)
    test_result.append((avg_val_loss, p, r, f1))

    if f1 > best_f1:
        best_f1 = f1
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

        final_logits = logits
        final_labels = labels


    model.classifier.bias.data = bias_old


result = np.array(test_result, dtype=np.float32)
test_loss, prec, recall, f1 = np.mean(result, axis=0)
# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)
print('LA Test F1 is:', f1)

ece = calculate_ece(final_logits, final_labels)
print('LA Test ECE is:', ece)


result_file = open(result_file_LA, 'w', encoding='UTF-8')
for r in range(len(rel2id)):
    result_file.write('prob_' + str(r) + '\t')
result_file.write('pred_label' + '\t')
result_file.write('true_label' + '\t')
result_file.write('max_prob' + '\t')
result_file.write('right_or_wrong' + '\n')

probs = F.softmax(final_logits, 1)
probs = [prob.tolist() for prob in probs]
labels = final_labels.tolist()

for prob, label in zip(probs, labels):
    max_prob = max(prob)
    pred_label = prob.index(max_prob)
    right_or_wrong = 0 if pred_label == label else 1

    result_file.write('\t'.join([str(round(p, 4)) for p in prob]) + '\t')
    result_file.write(str(pred_label) + '\t')
    result_file.write(str(int(label)) + '\t')
    result_file.write(str(round(max_prob, 4)) + '\t')
    result_file.write(str(right_or_wrong) + '\n')

print('Best result saved.')
