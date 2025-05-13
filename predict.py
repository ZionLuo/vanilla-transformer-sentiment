from textprocess_dataloader import *
from embedding_encoder import *
import pandas as pd

test_file = r'your test file'
df_test = pd.read_csv(test_file)

checkpoint = torch.load(r'transformer_model.pth')
config = checkpoint['config']
word_to_idx = checkpoint['word_to_idx']
vocab_size = len(word_to_idx)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
device = config['device']

test_croups = df_test['comment_text'].tolist()


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text):
    return text.lower().split()


def sentences_to_indices(word_to_idx,processed_corpus):
    unk_index = word_to_idx.get('<unk>')
    return [
        [word_to_idx.get(w, unk_index) for w in sentence]
        for sentence in processed_corpus
    ]

def load_model(config, vocab_size, device):
    embedding = TransformerEmbeddings(vocab_size, config['d_model']).to(device)
    encoder = TransformerEncoder(
        num_layers=config['num_encoder_layers'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    return embedding,encoder,word_to_idx


target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
cleaned_croups = [clean_text(text) for text in test_croups]
tokenized_cleaned_croups = [tokenize(text) for text in cleaned_croups]
embedding, encoder, word_to_idx = load_model(config, vocab_size, device)
start_indices = [[word_to_idx.get(word, word_to_idx['<unk>']) for word in sentence] for sentence in
                 tokenized_cleaned_croups]

data = PredictTextDataset(start_indices, unk_idx=word_to_idx['<unk>'])
dataloader = DataLoader(
    data,
    batch_size=config['batch_size'],
    collate_fn=predict_collate_fn,
    shuffle=False,
    drop_last=False)

all_preds = []
total = 0
for batch_inputs, lengths in dataloader:
    B = batch_inputs.size(0)
    total += B
    batch_inputs = batch_inputs.to(config['device'])
    src = batch_inputs.transpose(0, 1)
    emb = embedding(src)  # (B, L, D)
    logits = encoder(emb)  # (B, 6)
    all_preds.extend(logits.cpu().tolist())

submission_data = pd.DataFrame(all_preds, columns=target_columns)
submission_data['id'] = df_test['id'].values
submission_data = submission_data[['id'] + target_columns]

print(submission_data[:5])
#submission_file = 'your submission file'
#submission_data.to_csv(submission_file, index=False)