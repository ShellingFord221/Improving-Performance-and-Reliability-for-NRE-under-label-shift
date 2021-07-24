import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

from utils import data_prepare, format_time
import os
import time, json

from bert import BertForSequenceClassificationUserDefined
from calibration.built_in import TuneMaxThres, TuneEntropyThres, TuneMaxThres2, TuneEntropyThres2, TuneVarianceThres



CUDA = 0
DATASET = 'KBP'  
# NUM_LABELS = 7   # 19,42
MAX_LENGTH = 100
BATCH_SIZE = 32
save_path = './bert_' + DATASET + '.zip'
# save_path = './bert_' + DATASET + '.pt'
result_file = './' + DATASET + '.txt'



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


print('Loading model ...')
model = BertForSequenceClassificationUserDefined.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels=NUM_LABELS,  # The number of output labels--2 for binary classification.
    # You can increase this for multi-class tasks.
    output_attentions=False,  # Whether the model returns attentions weights.
    output_hidden_states=False,  # Whether the model returns all hidden-states.
)
model.load_state_dict(torch.load(save_path))
model.cuda()



print('Testing model ...')

t0 = time.time()
result_write = open(result_file, 'w', encoding='UTF-8')

_, _, MaxF1 = TuneMaxThres(model, test_dataloader)
result_write.write('Max threshold F1 is: ' + str(round(MaxF1, 4)) + '\n')
print('Max threshold F1 is:', round(MaxF1, 4))

_, _, EntF1 = TuneEntropyThres(model, test_dataloader)
result_write.write('Ent threshold F1 is: ' + str(round(EntF1, 4)) + '\n')
print('Ent threshold F1 is:', round(EntF1, 4))


result_write.write('\n')
print()


_, _, MaxMCDF1 = TuneMaxThres2(model, test_dataloader, NUM_LABELS, 30)
result_write.write('Max MCD threshold F1 is: ' + str(round(MaxMCDF1, 4)) + '\n')
print('Max MCD threshold F1 is:', round(MaxMCDF1, 4))

_, _, EntMCDF1 = TuneEntropyThres2(model, test_dataloader, NUM_LABELS, 30)
result_write.write('Ent MCD threshold F1 is: ' + str(round(EntMCDF1, 4)) + '\n')
print('Ent MCD threshold F1 is:', round(EntMCDF1, 4))

_, _, SVF1 = TuneVarianceThres(model, test_dataloader, NUM_LABELS, 30)
result_write.write('SV threshold F1 is:' + str(round(SVF1, 4)) + '\n')
print('SV threshold F1 is:', round(SVF1, 4))


result_write.close()


# Measure how long the validation run took.
validation_time = format_time(time.time() - t0)
print('Elapsed time', validation_time)

