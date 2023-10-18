import os
import torch
from torchtext.legacy import data

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


from konlpy.tag import Mecab
mecab = Mecab()

TEXT = data.Field(tokenize=mecab.morphs, include_lengths = True)
LABEL = data.LabelField(dtype = torch.float)


fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
# dictionary 형식은 {csv컬럼명 : (데이터 컬럼명, Field이름)}


train_data, test_data = data.TabularDataset.splits(
                            path = 'data',
                            train = 'train_data.csv',
                            test = 'test_data.csv',
                            format = 'csv',
                            fields = fields,  
)


import random

train_data, valid_data = train_data.split(random_state=random.seed(SEED))


MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data,
                max_size = MAX_VOCAB_SIZE,
                vectors = 'fasttext.simple.300d',
                unk_init = torch.Tensor.normal_)

LABEL.build_vocab(train_data)


print(len(TEXT.vocab))


print(LABEL.vocab.stoi)


print(vars(train_data.examples[0]))


for i in range(len(train_data)):
    if len(train_data.examples[i].text)==0:print(i)


BATCH_SIZE = 64

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')    # cuda & torch & torchtext version error

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size = BATCH_SIZE,
    #sort_within_batch = True,
    sort_key = lambda x: len(x.text),
    sort_within_batch = True,
    device = device)


print(next(iter(train_iterator)).text)


import torch.nn as nn


emb = nn.Embedding(3,5,padding_idx=1)
test = torch.tensor([0,1,2])


print(emb(test))    # padding_idx 에 해당하는 벡터는 0 이다


def print_shape(name, data):
    print(f'{name} has shape {data.shape}')


class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                n_layers, bidirectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        # text = [sent_len, batch_size]
        #print_shape('text',text)
        embedded = self.dropout(self.embedding(text))
        # embedded = [sent_len, batch_size, emb_dim]
        #print_shape('embedded', embedded)
        
        # pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        #print_shape('packed_output', packed_output)
        #print_shape('hidden', hidden)
        #print_shape('cell', cell)
        
        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        #print_shape('output', output)
        #print_shape('output_lengths', output_lengths)        
        
        # output = [sent_len, batch_size, hi_dim * num_directions]
        # output over padding tokens are zero tensors
        # hidden = [num_layers * num_directions, batch_size, hid_dim]
        # cell = [num_layers * num_directions, batch_size, hid_dim]
        
        # concat the final forward and backward hidden layers
        # and apply dropout
        
        #print_shape('hidden[-2,:,:]', hidden[-2,:,:])
        #print_shape('hidden[-1,:,:]', hidden[-1,:,:])
        #cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        #print_shape('cat', cat)

        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        #print_shape('hidden', hidden)
        # hidden = [batch_size, hid_dim * num_directions]
        
        res = self.fc(hidden)
        #print_shape('res', res)
        return res


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300 # fasttext dim과 동일하게 설정
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]


model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'이 모델은 {count_parameters(model):,} 개의 파라미터를 가지고 있다.')


pretrained_embeddings = TEXT.vocab.vectors

print(pretrained_embeddings.shape)


print(model.embedding.weight.data.shape)


print(model.embedding.weight.data.copy_(pretrained_embeddings)) # copy_ 메서드는 인수를 현재 모델의 웨이트에 복사함


UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
print(UNK_IDX, PAD_IDX)


model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

print(model.embedding.weight.data)


import torch.optim as optim

optimizer = optim.Adam(model.parameters())


criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds==y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        text, text_lengths = batch.text
        predictions = model(text, text_lengths).squeeze(1)
        #print_shape('predictions',predictions)
        
        loss = criterion(predictions, batch.label)
        #print_shape('loss',loss)
        
        acc = binary_accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)

            loss = criterion(predictions, batch.label)#.squeeze(0))
            acc = binary_accuracy(predictions, batch.label)#.squeeze(0))

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 5
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'sentiment_analysis_model.pt')
        
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


model.load_state_dict(torch.load('sentiment_analysis_model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


N_EPOCHS = 5

for epoch in range(N_EPOCHS):
    start_time = time.time()
    
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'sentiment_analysis_model.pt')
        
    print(f'Epoch: {epoch+6:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


model.load_state_dict(torch.load('sentiment_analysis_model.pt'))
test_loss, test_acc = evaluate(model, test_iterator, criterion)
print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


from konlpy.tag import Mecab
mecab = Mecab()


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = [tok for tok in mecab.morphs(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1) # 배치 
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(model(tensor, length_tensor))
    return prediction.item()


print(predict_sentiment(model, "이번 선거는 이재명이 이길거다. 이재명 최고!! 개혁의딸 화이팅"))
print(predict_sentiment(model, "이번 선거는 정의의 사도 윤석열이 무조건 이기지. 윤석열 화이팅!"))
print(predict_sentiment(model, "무상급식 오세훈은 사퇴하고 감옥가서 무상급식이나 얻어 먹어라!!"))
print(predict_sentiment(model, "지역구 버리고 출마한 송영길은 이번 선거 못 이기지. 두고 보자구??"))
