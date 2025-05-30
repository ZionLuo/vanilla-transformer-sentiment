import re
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

class TextProcessor:
    def __init__(self, croups):
        self.croups = croups
        self.vocab = None
        self.vocab_size = 0
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.processed_corpus = []

    def clean_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text):
        return text.lower().split()

    def build_vocab(self, top_k=1000):
        cleaned = [self.tokenize(self.clean_text(line))
                   for line in self.croups]
        if not cleaned:
            raise ValueError("清洗后的语料库为空")

        word_freq = Counter(word for sentence in cleaned for word in sentence)
        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']

        if len(special_tokens) >= top_k:
            raise ValueError(f"top_k必须至少为{len(special_tokens)}")

        # 构建词汇表
        top_words = [w for w, _ in word_freq.most_common(top_k - len(special_tokens))
                     if w not in special_tokens]
        vocab_list = special_tokens + top_words

        self.vocab = set(vocab_list)
        self.vocab_size = len(vocab_list)
        self.word_to_idx = {w: i for i, w in enumerate(vocab_list)}
        self.idx_to_word = {i: w for i, w in enumerate(vocab_list)}

        # 处理语料
        self.processed_corpus = []
        for sentence in cleaned:
            processed = ['<sos>'] + [w if w in self.vocab else '<unk>'
                                     for w in sentence] + ['<eos>']
            self.processed_corpus.append(processed)

        return self.word_to_idx

    def sentences_to_indices(self):
        unk_index = self.word_to_idx.get('<unk>')
        return [
            [self.word_to_idx.get(w, unk_index) for w in sentence]
            for sentence in self.processed_corpus
        ]


class TextDataset(Dataset):
    def __init__(self, indices, labels):
        assert len(indices) == len(labels)
        self.data = [torch.tensor(seq, dtype=torch.long) for seq in indices if len(seq)]
        self.labels = [torch.tensor(l, dtype=torch.long) for l in labels if len(l)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    seqs, labs = zip(*batch)
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = torch.max(lengths).item()
    B = len(seqs)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    labels = torch.stack(labs, dim=0)
    return padded, lengths, labels


class PredictTextDataset(Dataset):
    def __init__(self, indices, unk_idx):
        self.data = []
        for seq in indices:
            if len(seq) == 0:
                # 空句子至少保留一个 <unk> token
                self.data.append(torch.tensor([unk_idx], dtype=torch.long))
            else:
                self.data.append(torch.tensor(seq, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def predict_collate_fn(batch):
    seqs = [seq for seq in batch]
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = torch.max(lengths).item()
    B = len(seqs)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(seqs):
        padded[i, :len(seq)] = seq
    return padded, lengths