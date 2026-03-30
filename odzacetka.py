import json
import gzip

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.metrics import mean_absolute_error

import scipy.sparse as sp

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import os
from datetime import datetime


def load(path):
    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt", encoding="utf-8") as f:
        return json.load(f)

slovenian_public_holidays = [
    (1, 1),   # Novo leto
    (2, 1),   # Novo leto
    (8, 2),   # Prešernov dan
    (27, 4),  # Dan upora proti okupatorju
    (1, 5),   # Praznik dela
    (2, 5),   # Praznik dela
    (25, 6),  # Dan državnosti
    (15, 8),  # Marijino vnebovzetje (verski praznik, tudi dela prost dan)
    (31, 10), # Dan reformacije
    (1, 11),  # Dan spomina na mrtve
    (25, 12), # Božič
    (26, 12), # Dan samostojnosti in enotnosti
]
def mean_comments_dayinyear(data):
    day_comments = defaultdict(list)

    for a in data:
        day = datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").timetuple().tm_yday
        day_comments[day].append(a["n_comments"])

    mean_comments = {day: np.mean(comments) for day, comments in day_comments.items()}

    # Dodamo dni brez komentarjev (npr. prazne dneve)
    for i in range(1, 366):
        if i not in mean_comments:
            mean_comments[i] = 0.0  # ali np.nan, če želiš manjkajočo vrednost

    return mean_comments

def extract(data):
    # Dela prosti dnevi (številke dneva v letu, 1–365, ne prestopno leto)
    public_holiday_days = [1, 2, 39, 117, 121, 122, 176, 227, 304, 305, 359, 360]

    content = ["\n".join([a["title"]] + a["paragraphs"]) for a in data]
    topics = [[a.get("topics", "unknown")] for a in data]
    weekdays = [[datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").weekday()] for a in data]
    hours = [[datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").hour] for a in data]
    subtopics = [clanek["url"].split("/")[3:-2] for clanek in data]
    authors = [clanek.get("authors", "neznan") for clanek in data]
    days_after_holiday = [[(dy := datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").timetuple().tm_yday) - max([d for d in public_holiday_days if d <= dy], default=-999)] for a in data]
    return content, topics, weekdays, subtopics, hours, authors, days_after_holiday


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def dense_to_dataset(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1).float())


def torch_fit(dataset, lambda_=0, batch_size=1000, lr=0.1, epochs=20, collate_fn=None):
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn)
    # class FeedForwardNN(nn.Module):
    #     def __init__(self,inputs):
    #         super(FeedForwardNN, self).__init__()
    #         self.model = nn.Sequential(
    #             nn.Linear(inputs, 128),
    #             nn.ReLU(),
    #             nn.Dropout(p=0.3),
    #             nn.Linear(128, 1)
    #         )

    #     def forward(self, x):
    #         return self.model(x)

    # model = FeedForwardNN(dataset[0][0].shape[0])
    class Linear(nn.Module):
        def __init__(self, inputs):
            super().__init__()
            self.linear = nn.Linear(inputs, 1)

        def forward(self, x):
            return self.linear(x)

    model = Linear(dataset[0][0].shape[0])
    model = model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lambda_)

    def train(dataloader, model, loss_fn, optimizer):
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        correct /= size
        print(f"Avg loss: {test_loss:>8f} \n")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(loader, model, loss_fn, optimizer)
        if (t+1) % 5 == 0:
            test(loader, model, loss_fn)

    return model

def torch_predict(model, dataset):
    model.eval()
    x = dataset[:][0]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
    return pred.cpu().numpy().reshape(-1)


class RTVSlo:

    def fit(self, train_data: list):
        # self.mean_comments = mean_comments_dayinyear(train_data)
        # X_dayinyear = np.array([[self.mean_comments[datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").timetuple().tm_yday]] for a in train_data])

        content, onehot, weekdays, subtopics, hours, authors, day_after_holiday = extract(train_data)
        self.vectorizer = TfidfVectorizer(min_df=10, max_features=20000)  # avoid needing sparse data
        X_text = self.vectorizer.fit_transform(content)


        self.onehot = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        oh = self.onehot.fit_transform(onehot)

        self.onehot_days = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        X_days = self.onehot_days.fit_transform(weekdays)

        self.onehot_subtopics = MultiLabelBinarizer()
        X_subtopics = self.onehot_subtopics.fit_transform(subtopics)

        self.onehot_hours = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        X_hours = self.onehot_hours.fit_transform(hours)

        self.onehot_authors = MultiLabelBinarizer()
        X_authors = self.onehot_authors.fit_transform(authors)

        X_day_after = sp.csr_matrix(np.array(day_after_holiday, dtype=np.float32))


        X = sp.hstack([X_text, oh, X_days, X_subtopics, X_hours, X_authors]).astype(np.float32)
        X = X.toarray()
        y = np.array([d['n_comments'] for d in train_data])
        y = np.power(y, 0.45)
        self.torch_model = torch_fit(dense_to_dataset(X, y), lambda_=0.0001)

    def predict(self, test_data: list):
        content, onehot, weekdays, subtopics, hours, authors, day_after_holiday = extract(test_data)

        X_text = self.vectorizer.transform(content)
        oh = self.onehot.transform(onehot)
        X_days = self.onehot_days.transform(weekdays)
        X_subtopics = self.onehot_subtopics.transform(subtopics)
        X_hours = self.onehot_hours.transform(hours)
        X_authors = self.onehot_authors.transform(authors)
        X_day_after = sp.csr_matrix(np.array(day_after_holiday, dtype=np.float32))
        # X_dayinyear = np.array([[self.mean_comments.get(datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").timetuple().tm_yday, 0)] for a in test_data])

        X = sp.hstack([X_text, oh, X_days, X_subtopics, X_hours, X_authors]).astype(np.float32)
        X = X.toarray()
        y =  torch_predict(self.torch_model, dense_to_dataset(X, np.zeros(len(X))))
        y = np.clip(y, 0, None)
        y = np.power(y, 1/0.45)
        return y

def test_mae(val_data, model):
    preds = model.predict(val_data)
    y = np.array([d['n_comments'] for d in val_data])
    mae = mean_absolute_error(y, preds)
    print(f"MAE: {mae:.2f}")
    return mae
if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    # this shows how your solution should be called
    train_data = load("data/rtvslo_train.json")
    test = load("data/rtvslo_test.json")
    n = len(train_data)
    train_data, val_data = train_data[:(n*4)//5], train_data[(n*4)//5:]
    m = RTVSlo()
    m.fit(train_data)
    test_mae(val_data, m)
    p = m.predict(test)

    np.savetxt('example.txt', p, fmt='%f')
