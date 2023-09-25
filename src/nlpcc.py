import argparse
import os
import time

import jieba

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchtext import vocab
from torchtext.transforms import LabelToIndex

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


parser = argparse.ArgumentParser()

parser.add_argument("--train_set", type=str, default="dataset/NLPCC2017/train.txt")
parser.add_argument("--test_set", type=str, default="dataset/NLPCC2017/test.txt")
parser.add_argument("--stopwords", type=str, default="dataset/stopwords.txt")
parser.add_argument("--user_dict", type=str, default="dataset/dictionary.txt")

parser.add_argument("--embedding_dim", type=int, default=512)

parser.add_argument("--epoch", type=int, default=15)
parser.add_argument("--batch_size", type=int, default=512)

parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--beta1", type=float, default=0.9)
parser.add_argument("--beta2", type=float, default=0.99)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=1e-6)

parser.add_argument("--sample_interval", type=int, default=100)


opt = parser.parse_args()
print("{}".format(opt)[10:-1].replace(", ", "\n").replace("=", ": "))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataset(Dataset):
    def __init__(self, root, stopwords_path=None, user_dict=None, word_spliter=jieba.lcut_for_search) -> None:
        super().__init__()

        start_time = time.time()

        if stopwords_path is not None:
            with open(stopwords_path) as stopwords_file:
                self.stopwords = set(stopwords_file.readlines())
        else:
            self.stopwords = set()

        if user_dict is not None:
            jieba.load_userdict(user_dict)

        self.X = list()
        self.y = list()

        with open(root) as data_file:
            for example in data_file.readlines():
                y, X = example.strip().split("\t")
                X = [word for word in word_spliter("".join(X.split(" "))) if word not in self.stopwords]

                self.X.append(X)
                self.y.append(y)

        self.labels = set(self.y)
        self.label_to_index = LabelToIndex(self.labels, sort_names=True)
        self.index_to_label = { self.label_to_index(label) : label for label in self.labels }
        self.label_to_index = { label : self.label_to_index(label) for label in self.labels }

        self.vocab = vocab.build_vocab_from_iterator(self.X, 2, "<UNK>")
        self.vocab.set_default_index(0)

        print("Cost {}s loading dataset.".format(time.time() - start_time))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.label_to_index[self.y[index]], self.X[index]

train_set = TextDataset(opt.train_set, opt.stopwords, opt.user_dict)
test_set = TextDataset(opt.test_set, opt.stopwords, opt.user_dict)


def collate_batch(batch):
    """
    Process data in order to build data loader. Referencing to https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#generate-data-batch-and-iterator.
    """
    label_list, text_list, offsets = [], [], [0]
    for y, X in batch:
        label_list.append(y)
        processed_text = torch.tensor(train_set.vocab(X), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
    text_list = torch.cat(text_list).to(device)
    return label_list, text_list, offsets


train_loader = DataLoader(train_set, opt.batch_size, True, collate_fn=collate_batch)
test_loader = DataLoader(test_set, opt.batch_size, False, collate_fn=collate_batch)


class FastText(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()

        self.embedding = nn.EmbeddingBag(len(train_set.vocab), embedding_dim)
        self.a = nn.LeakyReLU(0.1, True)
        self.pooling = nn.AvgPool1d(kernel_size=2, stride=2)
        self.fc = nn.Linear(embedding_dim // 2, len(train_set.labels))

    def forward(self, text, offset):
        return self.fc(self.pooling(self.a(self.embedding(text, offset))))


model = FastText(opt.embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
criterion = nn.CrossEntropyLoss().to(device)
total_accu = None

def train(model, data_loader, criterion, optimizer):
    """
    Train the model on training set in a single epoch.
    """
    model.train()
    total_acc, total_count = 0, 0

    for index, (y, X, offsets) in enumerate(data_loader):
        optimizer.zero_grad()
        pred = model(X, offsets)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_acc += (pred.argmax(1) == y).sum().item()
        total_count += y.size(0)
        if index % opt.sample_interval == 0 and index > 0:
            print(
                "| {:5d}/{:5d} batches | accuracy {:8.3f} ".format(
                    index, len(data_loader), total_acc / total_count
                )
            )
            total_acc, total_count = 0, 0


def test(model, data_loader):
    """
    Evaluate the model on test set.
    """
    model.eval()
    y_true: list = list()
    y_pred: list = list()

    with torch.no_grad():
        for y, X, offsets in data_loader:
            pred = model(X, offsets)
            y_true += [ y[i].item() for i in range(y.shape[0]) ]
            y_pred += [ pred.argmax(1)[i].item() for i in range(pred.argmax(1).shape[0]) ]

    # p: dict = dict()
    # r: dict = dict()

    # for index, label in enumerate(train_set.labels):
    #     p[label] = precision_score([1 if y == index else 0 for y in y_true], [1 if y == index else 0 for y in y_pred], average="binary")
    #     r[label] = recall_score([1 if y == index else 0 for y in y_true], [1 if y == index else 0 for y in y_pred], average="binary")

    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred, average="macro"), recall_score(y_true, y_pred, average="macro"), f1_score(y_true, y_pred, average="macro")


print("\nTraining start =========================================================")
for epoch in range(1, opt.epoch + 1):
    epoch_start_time = time.time()
    train(model, train_loader, criterion, optimizer)
    avg_acc, marco_p, macro_r, f1 = test(model, test_loader)
    if total_accu is not None and total_accu > avg_acc:
        scheduler.step()
    else:
        total_accu = avg_acc
    print("-" * 59)
    print(
        "| end of epoch {:3d} | time: {:5.2f}s | Accuracy {:8.3f} | Macro-P {:8.3f} | Macro-R {:8.3f} | f1 score {:8.3f}".format(
            epoch, time.time() - epoch_start_time, avg_acc, marco_p, macro_r, f1
        )
    )
    print("-" * 59)


model.eval()
while True:
    sentence = input("Enter a news headline (or `.` to exit): ")
    if sentence == ".":
        break
    sentence = [word for word in jieba.lcut_for_search(sentence) if len(word) > 1 and word not in train_set.stopwords]
    pred = model(torch.tensor(train_set.vocab(sentence)).to(device), torch.tensor([0]).to(device))
    print("News category: {}".format(train_set.index_to_label[pred.argmax(1)[0].item()]))
