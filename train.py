from transformers import AdamW
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import random
import numpy as np
import os
import time, json

from bert import BertForSequenceClassificationUserDefined
from utils import data_prepare, format_time, score



CUDA = 0
DATASET = 'KBP'   
# NUM_LABELS = 7   # 19,42
MAX_LENGTH = 100
BATCH_SIZE = 32
# MINI_BATCH_SIZE = 10000
LR = 2e-5
EPS = 1e-8
EPOCHS = 30


PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
ner2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NATIONALITY': 2, 'SET': 3, 'ORDINAL': 4, 'ORGANIZATION': 5, 'MONEY': 6, 'PERCENT': 7, 'URL': 8, 'DURATION': 9, 'PERSON': 10, 'CITY': 11, 'CRIMINAL_CHARGE': 12, 'DATE': 13, 'TIME': 14, 'NUMBER': 15, 'STATE_OR_PROVINCE': 16, 'RELIGION': 17, 'MISC': 18, 'CAUSE_OF_DEATH': 19, 'LOCATION': 20, 'TITLE': 21, 'O': 22, 'COUNTRY': 23, 'IDEOLOGY': 24}
pos2id = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46}


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
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
rel2id = json.load(open('./'+DATASET+'/relation2id.json', 'r'))
NUM_LABELS = len(rel2id)


print('Preparing data ...')
# Combine the training inputs into a TensorDataset.
t1, t2, t3, t4, t5 = data_prepare('train.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
train_dataset = TensorDataset(t1, t2, t3, t4, t5)

d1, d2, d3, d4, d5 = data_prepare('dev.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
dev_dataset = TensorDataset(d1, d2, d3, d4, d5)

# s1, s2, s3, s4, s5 = data_prepare('test.json', DATASET, MAX_LENGTH, rel2id, tokenizer)
# test_dataset = TensorDataset(s1, s2, s3, s4, s5)

print('Data prepared.')



train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler=RandomSampler(train_dataset),  # Select batches randomly
            batch_size=BATCH_SIZE  # Trains with this batch size.
        )
dev_dataloader = DataLoader(
            dev_dataset,  # The training samples.
            sampler=SequentialSampler(dev_dataset),  # Select batches randomly
            batch_size=BATCH_SIZE  # Trains with this batch size.
        )
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

model.cuda()


optimizer = AdamW(model.parameters(),
                  lr=LR,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=EPS  # args.adam_epsilon  - default is 1e-8.
                )


# Total number of training steps is [number of batches] x [number of epochs].
total_steps = len(train_dataloader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,  # Default value in run_glue.py
                                            num_training_steps=total_steps)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)




# ========================================
#               Training
# ========================================


training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

best_f1 = 0

# For each epoch...
for epoch_i in range(0, EPOCHS):

    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, EPOCHS))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)

            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))


        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_e1_pos = batch[3].to(device)
        b_e2_pos = batch[4].to(device)


        model.zero_grad()

        loss, logits = model(b_input_ids,
                             token_type_ids=None,
                             attention_mask=b_input_mask,
                             labels=b_labels,
                             e1_pos=b_e1_pos,
                             e2_pos=b_e2_pos)


        total_train_loss += loss.item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)

    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))


    print('')
    print('Validating ...')

    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    all_prediction = np.array([])
    all_ground_truth = np.array([])

    # Evaluate data for one epoch
    for batch in dev_dataloader:
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
    avg_val_loss = total_eval_loss / len(dev_dataloader)

    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)

    _, _, f1 = score(all_ground_truth, all_prediction)

    if f1 > best_f1:
        best_f1 = f1
        print('Best dev F1:', best_f1)
        torch.save(model.state_dict(), './bert_' + DATASET + '.pt')



print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
print("Best dev F1 is", best_f1)
