import torch
import torch.nn as nn


class CBOW(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)

        self.linear_1 = nn.Linear(emb_dim, hidden_1)
        self.activation_1 = nn.ReLU()

        self.linear_2 = nn.Linear(hidden_1, vocab_size)
        # self.activation_2 = nn.LogSoftmax(-1)  #skip, use CrossEntropyLoss()

    def forward(self, inp):
        embedding = self.emb(inp).sum(1)
        out = self.linear_1(embedding)
        out = self.activation_1(out)
        out = self.linear_2(out)
        return out


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, context_size, word_to_index):
        self.x = []
        self.y = []
        for i in range(context_size, len(vocab) - context_size):
            context = vocab[i - context_size: i + context_size + 1]
            target = context.pop(context_size)
            self.x.append(self.generate_context_vector(context, word_to_index))
            self.y.append(word_to_index[target])

    @staticmethod
    def generate_context_vector(words, word_to_index):
        indices = [word_to_index[w] for w in words]
        return torch.tensor(indices, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train(model, epochs, dataloader, loss_fn, optimizer):
    model.train()
    for i in range(epochs):
        train_loss = 0  # sum across batches
        for data, label in dataloader:
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.item()
        print(f"epoch {i + 1}/{epochs}, loss:", train_loss)


def test(model, test_dataset, index_to_word):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    model.eval()
    with torch.no_grad():
        for data, label in test_dataloader:
            out = model(data)
            given_words = [index_to_word[i.item()] for i in data.reshape(-1)]
            print(f"data: {given_words[: len(given_words) // 2]} [?] {given_words[len(given_words) // 2:]}")
            print(f'Predict: {index_to_word[torch.argmax(out[0]).item()]}\n')


if __name__ == '__main__':
    text = "We are about to study the idea of a computational process. Computational processes are abstract beings that inhabit computers. As they evolve, processes manipulate other abstract things called data. The evolution of a process is directed by a pattern of rules called a program. People create programs to direct processes. In effect, we conjure the spirits of the computer with our spells."

    vocab = set(text.split())

    vocab_size, emb_dim, hidden_1 = len(vocab), 50, 100
    context_size = 2

    lr = 0.001
    epochs = 100

    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for index, word in enumerate(vocab)}

    model = CBOW(vocab_size, emb_dim, hidden_1)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train
    train_dataset = CustomDataset(text.split(), context_size, word_to_index)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)
    train(model, epochs, train_dataloader, loss_fn, optimizer)

    # test
    context_test = "We are about to study the idea of a computational process".split()
    test_dataset = CustomDataset(context_test, context_size, word_to_index)
    test(model, test_dataset, index_to_word)

    # ...
    # epoch 100 / 100, loss: 0.047268493101000786
    #
    # data: ['We', 'are'][?] ['to', 'study']
    # Predict: about
    # ...
