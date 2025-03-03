import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from sklearn.preprocessing import OneHotEncoder


d_size = 10  # embedding dim, The bigger the dimension the more examples you need for training
device = "mps"
epochs = 100
lr = 1e-3
context_size = 3
batch_size = 128


class ReviewDataset(Dataset):
    def __init__(self, review_context, review_target):
        super(ReviewDataset, self).__init__()

        self.context = torch.tensor(review_context, dtype=torch.long)
        self.targets = torch.tensor(review_target, dtype=torch.long)

    def __len__(self):
        return len(self.context)

    def __getitem__(self, index):
        return self.context[index], self.targets[index]


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, d_size):
        super(CBOW, self).__init__()

        # This can also be a linear layer (multiplying a one hot encoded vector with the w matrix or just grabbing the correct index and doing backward on that index only) but the embedding saves us the work by storing the embeddings and we can get them back using indices
        # self.W = torch.nn.Linear(vocab_size, d_size)
        self.W_embedding = torch.nn.Embedding(vocab_size, d_size)
        self.linear1 = torch.nn.Linear(d_size * context_size * 2, 256) # we want the same thing as the sum of the embeddings (all of this context embeddings get combined)
        self.U = torch.nn.Linear(d_size, vocab_size)

    def forward(self, x):
        # project one hot encoding vectors to their embeddings
        x = self.W_embedding(x)
        x = x.sum(axis=1)  # sum the embeddings
        # x = x.mean(axis=1)

        # x = F.relu(self.linear1(x.view(x.shape[0], -1)))

        x = self.U(x)  # project to the Vocab space (the output embedding)

        # Because we want to predict the output with a softmax (from all the embeddings we want to predict what should be the output for the context words).
        # But we don't take into account the global-co ocurence statistics (so we wouldn't know that dogs and cats are really highly related )

        return x


def get_corpus(path):
    text_small = pd.read_csv(path)

    review_corpus = [list(gensim.utils.tokenize(text_small["text"][i]))
                     for i in range(len(text_small))]

    vocab = set()

    for i in review_corpus:
        for j in i:
            vocab.add(j)

    word2idx = {word: idx for idx, word in enumerate(set(vocab))}

    return review_corpus, list(vocab), word2idx


def convert_to_one_hot_encoded_windows(sentences, word2idx, windows_size=2):
    window_context = []
    window_target = []

    for sentence in sentences:
        for i in range(windows_size, len(sentence) - windows_size):
            target_word = sentence[i]
            target_word_index = word2idx[target_word]

            prefix = sentence[i - windows_size:i]
            suffix = sentence[i + 1: i + windows_size + 1]

            prefix_index = [word2idx[word] for word in prefix]
            suffix_index = [word2idx[word] for word in suffix]

            context = []
            context.extend(prefix_index)
            context.extend(suffix_index)

            window_context.append(context)
            window_target.append(target_word_index)

    return window_context, window_target


def train_model(model, train_dataloader, test_dataloader):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion_loss = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        
        model.train()
        for idx, batch in enumerate(train_dataloader):
            if idx % 1000 == 0:
                print(f"{idx}/{len(train_dataloader)}")
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            # print(y[0], F.softmax(pred[0], dim=-1).argmax())
            loss = criterion_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Training Loss: {total_loss / len(train_dataloader)}")

        model.eval()

        total_loss_test = 0
        for idx, batch in enumerate(test_dataloader):
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x) # logits
            loss = criterion_loss(pred, y)

            total_loss_test += loss.item()

        print(
            f"Epoch {epoch+1}/{epochs}, Test Loss: {total_loss_test / len(test_dataloader)}")


def cosine_similarity(model, target_word, word2idx, vocab):
    model.eval()

    word_embeddings = model.W_embedding

    # Get index of target word
    if target_word not in word2idx:
        print(f"Word '{target_word}' not found in vocabulary!")
        return

    target_idx = torch.tensor(word2idx[target_word]).to(device)

    target_embedding = word_embeddings(target_idx)

    cosine_similarities = F.cosine_similarity(
        target_embedding, word_embeddings.weight.data)

    top_indices = torch.argsort(cosine_similarities, descending=True)[
        1:6]  # Exclude self-match

    print(f"Top similar words to '{target_word}':")
    for idx in top_indices:
        print(
            f"{vocab[idx]} (Cosine Similarity: {cosine_similarities[idx].item():.4f})")


def main():
    review_corpus, vocab, word2idx = get_corpus("data/small_corpus.csv")

    vocab = np.array(vocab).reshape(-1, 1)

    vocab_size = len(vocab)

    # encoder = OneHotEncoder(sparse_output=False, dtype=np.float32)
    # one_hot_vocab = encoder.fit_transform(vocab)

    review_context, review_target = convert_to_one_hot_encoded_windows(
        review_corpus, word2idx, context_size)
    
    print(f"Review size {len(review_context)} and vocab size {len(vocab)}")

    print("created one hot encoded")

    review_dataset = ReviewDataset(review_context, review_target)
    train_dataset, test_dataset = random_split(review_dataset, [0.8, 0.2])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("created dataloader")

    model = CBOW(vocab_size, d_size)
    model = model.to(device)

    print("started training")
    train_model(model, train_dataloader, test_dataloader)

    cosine_similarity(model, "word", word2idx, vocab)


if __name__ == "__main__":
    main()
