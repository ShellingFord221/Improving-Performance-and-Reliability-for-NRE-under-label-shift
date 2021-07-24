import torch
from torch.nn import functional as F
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

from utils import data_prepare, format_time, score, calculate_ece
import os
import time, json
import numpy as np

from bert import BertForSequenceClassificationUserDefined



CUDA = 0
DATASET = 'KBP'  
# NUM_LABELS = 7   # 19,42
MAX_LENGTH = 100
BATCH_SIZE = 32
save_path = './bert_' + DATASET + '.zip'
# save_path = './bert_' + DATASET + '.pt'
result_file_original = './bert_' + DATASET + '_orig.txt'



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
s1, s2, s3, s4, s5, _, _ = data_prepare('test.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
test_dataset = TensorDataset(s1, s2, s3, s4, s5)

test_dataloader = DataLoader(
            test_dataset,  # The training samples.
            sampler=SequentialSampler(test_dataset),  # Select batches randomly
            batch_size=BATCH_SIZE  # Trains with this batch size.
        )



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

# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)

_, _, f1 = score(all_ground_truth, all_prediction)

print('Test F1 is:', f1)


logits = torch.cat(logits_list).cuda()
labels = torch.cat(labels_list).cuda()
ece = calculate_ece(logits, labels)
print('Test ECE is:', ece)



result_file = open(result_file_original, 'w', encoding='UTF-8')
for r in range(len(rel2id)):
    result_file.write('prob_' + str(r) + '\t')
result_file.write('pred_label' + '\t')
result_file.write('true_label' + '\t')
result_file.write('max_prob' + '\t')
result_file.write('right_or_wrong' + '\n')

probs = F.softmax(logits, 1)
probs = [prob.tolist() for prob in probs]
labels = labels.tolist()

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
