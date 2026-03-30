import json
import gzip
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import scipy.sparse as sp

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import os
import time
import torch
from sklearn.linear_model import Ridge
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer
import pickle
# Naprava (GPU če je na voljo)
device = "cuda" if torch.cuda.is_available() else "cpu"

def load(path):
    open_func = gzip.open if path.endswith(".gz") else open
    with open_func(path, "rt", encoding="utf-8") as f:
        return json.load(f)

model_name = "Helsinki-NLP/opus-mt-tc-bible-big-sla-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

def translate(texts):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    gen = model.generate(**batch)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in gen]


def dodaj_angleski_prevod(data):
    print("Prevodim besedilo v angleščino...")
    for clanek in tqdm(data):
        if "anglesko_besedilo" not in clanek:
            clanek["anglesko_besedilo"] = " ".join(translate(clanek["paragraphs"]))
        if "angleski_naslov" not in clanek:
            clanek["angleski_naslov"] = translate([clanek["title"]])
    return data


import stanza
from tqdm import tqdm 
stanza.download('sl')
nlp = stanza.Pipeline(lang='sl', processors='tokenize,lemma', use_gpu=True)

def lematiziraj_besedilo(besedilo):
    """Lematizira eno besedilo."""
    doc = nlp(besedilo)
    lematizirane_besede = [word.lemma for sentence in doc.sentences for word in sentence.words]
    return " ".join(lematizirane_besede)

def lemmatized_text(data):
    print("Lematiziram besedilo...")
    for clanek in tqdm(data):
        if "lemmatized_text" not in clanek:
            besedilo = " ".join(clanek["paragraphs"])
            clanek["lemmatized_text"] = lematiziraj_besedilo(besedilo)

def pop_za_ta_topic(data):
    dict = {}
    dict_clankov = {}
    for clanek in data:
        for subtopic in clanek["topics"]:
            if subtopic not in dict:
                dict[subtopic] = 0
            if subtopic not in dict_clankov:
                dict_clankov[subtopic] = 0
            dict_clankov[subtopic] += clanek["n_comments"]
            dict[subtopic] += 1
    dict_pop_za_ta_topic = {}
    for clanek in data:
        topic = clanek["topics"][0]
        dict_pop_za_ta_topic[topic] = dict_clankov[topic] / dict[topic]
    return dict_pop_za_ta_topic

def pop_za_tega_avtorja(data):
    dict = {}
    dict_num_clankov = {}
    for clanek in data:
        avtor = clanek.get("authors",["neznan"])[0]
        if avtor not in dict:
            dict[avtor] = 0
        dict[avtor] += clanek["n_comments"]
        if avtor not in dict_num_clankov:
            dict_num_clankov[avtor] = 0
        dict_num_clankov[avtor] += 1
    dict_pop_za_tega_avtorja = {}
    for clanek in data:
        avtor = clanek.get("authors",["neznan"])[0]
        dict_pop_za_tega_avtorja[avtor] = dict[avtor] / dict_num_clankov[avtor]
    return dict_pop_za_tega_avtorja

def extract(data):
    sin_weekdan = []
    cos_weekdan = []
    sin_dayofyear = []
    cos_dayofyear = []
    sin_hour = []
    cos_hour = []
    for clanek in data:
        avtor = clanek.get("authors",["neznan"])[0]
        clanek["author"] = avtor
        topics = clanek["url"].split("/")[3:-2]
        clanek["topics"] = " ".join(topics)
        # Pretvorba datuma v število dni od začetka leta
        datum = datetime.strptime(clanek["date"], "%Y-%m-%dT%H:%M:%S")
        clanek["date_days"] = (datum - datetime(datum.year, 1, 1)).days
        #print(f"Datum: {datum}, Dni od začetka leta: {clanek['date_days']}, dan v tednu: {datum.weekday()}, ura: {datum.hour}")
        clanek["embeddings"] = np.array(clanek["embeddings"])
        sin_weekdan.append(np.sin(2 * np.pi * datum.weekday() / 7))
        cos_weekdan.append(np.cos(2 * np.pi * datum.weekday() / 7))
        sin_dayofyear.append(np.sin(2 * np.pi * clanek["date_days"] / 365))
        cos_dayofyear.append(np.cos(2 * np.pi * clanek["date_days"] / 365))
        sin_hour.append(np.sin(2 * np.pi * datum.hour / 24))
        cos_hour.append(np.cos(2 * np.pi * datum.hour / 24))


    text = [a["lemmatized_text"] for a in data]
    titles = [a["title"] for a in data]
    content = [a["embeddings"] for a in data]
    topics = [[a.get("topics", "unknown")] for a in data]
    weekdays = [[datetime.strptime(a["date"], "%Y-%m-%dT%H:%M:%S").weekday()] for a in data]
    authors = [[a.get("authors", ["unknown"])[0]] for a in data]   # Dodajanje avtorja
    dates = [[a.get("date_days", 0)] for a in data]  # Dodajanje dni od začetka leta
    
    return text, titles, content, topics, weekdays, authors, dates , sin_weekdan, cos_weekdan, sin_dayofyear, cos_dayofyear, sin_hour, cos_hour

def dense_to_dataset(X, y):
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1).float())

def torch_fit(dataset, lambda_=0, batch_size=30000, lr=0.01, epochs=100, collate_fn=None):
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
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

    def train(dataloader, model, loss_fn, optimizer, lambda_):
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Dodaj L1 regularizacijo
            # l1_lambda = lambda_ * sum(p.abs().sum() for p in model.parameters())
            # loss += l1_lambda

            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

    def test(dataloader, model, loss_fn):
        num_batches = len(dataloader)
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
        test_loss /= num_batches
        print(f"Avg loss: {test_loss:.6f}\n")

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(loader, model, loss_fn, optimizer, lambda_)
        if (t + 1) % 5 == 0:
            test(loader, model, loss_fn)

    return model

def torch_predict(model, dataset):
    model.eval()
    x = dataset[:][0]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
    return pred.cpu().numpy().reshape(-1)

def sloberta_embedings(data, batch_size=16):
    def split_text(text, max_tokens=512):
        words = text.split()
        return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/sloberta")
    model = AutoModel.from_pretrained("EMBEDDIA/sloberta")
    model.to(device)
    vectors = []
    for article in tqdm(data):
        segments = split_text(" ".join(article["paragraphs"])+" "+article["title"])
        segment_vectors = []

        for segment in segments:
            tokens = tokenizer(segment, return_tensors="pt", truncation=True, max_length=512).to(device)  # Premakni na GPU
            output = model(**tokens)
            vector = torch.mean(output.last_hidden_state, dim=1).detach().cpu().numpy()  # Premakni rezultat nazaj na CPU
            segment_vectors.append(vector)

        # Povprečje vseh segmentov za končno vektorsko predstavitev članka
        article_vector = sum(segment_vectors) / len(segment_vectors)
        vectors.append(article_vector)
    return vectors

def sentence_transformers_embedings(data):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    for article in tqdm(data):
        text = " ".join(article["paragraphs"]) + " " + article["title"]
        vector = model.encode(text, show_progress_bar=False)
        article["sentence_transformers_embedings"] = vector

def tfidf_embeddings(data):
    vectorizer = TfidfVectorizer(min_df=10, max_features=20000)
    for article in tqdm(data):
        text = " ".join(article["paragraphs"]) + " " + article["title"]
        vector = vectorizer.fit_transform([text]).toarray()
        article["tfidf_embeddings"] = vector[0]

class RTVSlo:

    def fit(self, train_data: list):
        text, titles, content, topics, weekdays, authors, dates, sin_weekdan, cos_weekdan, sin_dayofyear, cos_dayofyear, sin_hour, cos_hour = extract(train_data)

        self.vectorizer = TfidfVectorizer(min_df=10, max_features=20000)
        X_text_embeddings = self.vectorizer.fit_transform(text)
        
        self.vectorizer_for_titles = TfidfVectorizer(min_df=10, max_features=20000)
        X_titles = self.vectorizer_for_titles.fit_transform(titles)

        X_text = sp.csr_matrix(np.vstack(content))

        self.onehot_topics = OneHotEncoder(handle_unknown="ignore", max_categories=50)
        X_topics = self.onehot_topics.fit_transform(topics)

        self.onehot_days = OneHotEncoder(handle_unknown="ignore", categories='auto')
        X_days = self.onehot_days.fit_transform(weekdays)

        self.onehot_authors = OneHotEncoder(handle_unknown="ignore")
        X_authors = self.onehot_authors.fit_transform(authors)

        # Pretvorba datumov v numerično vrednost
        X_dates = np.array(dates, dtype=np.float32)

        # Sinus in kosinus funkcije za dneve v tednu, dneve v letu in ure
        X_sin_weekdan = np.array(sin_weekdan, dtype=np.float32).reshape(-1, 1)
        X_cos_weekdan = np.array(cos_weekdan, dtype=np.float32).reshape(-1, 1)
        X_sin_dayofyear = np.array(sin_dayofyear, dtype=np.float32).reshape(-1, 1)
        X_cos_dayofyear = np.array(cos_dayofyear, dtype=np.float32).reshape(-1, 1)
        X_sin_hour = np.array(sin_hour, dtype=np.float32).reshape(-1, 1)
        X_cos_hour = np.array(cos_hour, dtype=np.float32).reshape(-1, 1)

        # Poprečje za ta avtorja
        self.dict_pop_ta_tega_avtorja = pop_za_tega_avtorja(train_data)
        pop_ta_tega_avtorja_list = [self.dict_pop_ta_tega_avtorja[clanek["author"]] for clanek in train_data]
        X_pop_avtor = np.array(pop_ta_tega_avtorja_list, dtype=np.float32).reshape(-1, 1)
        
        #Poprečje za ta topic
        self.dict_pop_za_ta_topic = pop_za_ta_topic(train_data)
        pop_za_ta_topic_list = [self.dict_pop_za_ta_topic[clanek["topics"][0]] for clanek in train_data]
        X_pop_topic = np.array(pop_za_ta_topic_list, dtype=np.float32).reshape(-1, 1)

        # Združevanje vseh vhodov (besedilo, teme, dnevi, avtorji, datumi)
        #, X_sin_weekdan, X_cos_weekdan, X_sin_dayofyear, X_cos_dayofyear, X_sin_hour, X_cos_hour
        X = sp.hstack([X_text_embeddings , X_titles, X_authors, X_pop_topic, X_pop_avtor, X_topics, X_sin_weekdan, X_cos_weekdan, X_sin_dayofyear, X_cos_dayofyear, X_sin_hour, X_cos_hour ]).astype(np.float32).toarray()
        y = np.sqrt(np.array([d['n_comments'] for d in train_data]))

        self.ridge_model = Ridge(alpha=0.1)
        self.ridge_model.fit(X, y)

    def predict(self, test_data: list):

        text, titles, content, topics, weekdays, authors, dates,  sin_weekdan, cos_weekdan, sin_dayofyear, cos_dayofyear, sin_hour, cos_hour  = extract(test_data)

        X_text_embeddings = self.vectorizer.transform(text)
        X_titles = self.vectorizer_for_titles.transform(titles)
        X_text = sp.csr_matrix(np.vstack(content))
        X_topics = self.onehot_topics.transform(topics)
        X_days = self.onehot_days.transform(weekdays)
        X_authors = self.onehot_authors.transform(authors)
        X_dates = np.array(dates, dtype=np.float32)
        X_sin_weekdan = np.array(sin_weekdan, dtype=np.float32).reshape(-1, 1)
        X_cos_weekdan = np.array(cos_weekdan, dtype=np.float32).reshape(-1, 1)
        X_sin_dayofyear = np.array(sin_dayofyear, dtype=np.float32).reshape(-1, 1)
        X_cos_dayofyear = np.array(cos_dayofyear, dtype=np.float32).reshape(-1, 1)
        X_sin_hour = np.array(sin_hour, dtype=np.float32).reshape(-1, 1)
        X_cos_hour = np.array(cos_hour, dtype=np.float32).reshape(-1, 1)


        # Poprečje za ta avtorja
        pop_ta_tega_avtorja_list = [self.dict_pop_ta_tega_avtorja.get(clanek["author"],0) for clanek in test_data]
        X_pop_avtor = np.array(pop_ta_tega_avtorja_list, dtype=np.float32).reshape(-1, 1)

        #Poprečje za ta topic
        pop_za_ta_topic_list = [self.dict_pop_za_ta_topic[clanek["topics"][0]] for clanek in test_data]
        X_pop_topic = np.array(pop_za_ta_topic_list, dtype=np.float32).reshape(-1, 1)
        
        # Združevanje vseh vhodov (besedilo, teme, dnevi, avtorji, datumi)
        X = sp.hstack([X_text_embeddings ,X_titles, X_authors, X_pop_topic, X_pop_avtor, X_topics,X_sin_weekdan,X_cos_weekdan,X_sin_dayofyear,X_cos_dayofyear,X_sin_hour,X_cos_hour ]).astype(np.float32).toarray()

        #preds = torch_predict(self.torch_model, dense_to_dataset(X, np.zeros(len(X))))
        preds = self.ridge_model.predict(X)
        preds = np.square(preds)
        preds = np.clip(preds, 0, None)

        return preds

def test_mae(val_data, model):
    preds = model.predict(val_data)
    y = np.array([d['n_comments'] for d in val_data])
    mae = mean_absolute_error(y, preds)
    print(f"MAE: {mae:.2f}")
    return mae

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))

    train_file = "train.pkl"
    if os.path.exists(train_file):
        with open(train_file, "rb") as file:
            train = pickle.load(file)
        print(train[-1]["lemmatized_text"])
        if "sentence_transformers_embedings" not in train[0]:
            sentence_transformers_embedings(train)
            with open(train_file, "wb") as file:
                pickle.dump(train, file)
        if "lemmatized_text" not in train[0]:
            lemmatized_text(train)
            with open(train_file, "wb") as file:
                pickle.dump(train, file)
    else:
        train = load("data/rtvslo_train.json")
        emmbedings = sloberta_embedings(train)
        dodaj_angleski_prevod(train)
        for i, article in enumerate(train):
            article["embeddings"] = emmbedings[i]
        with open(train_file, "wb") as file:
            pickle.dump(train, file)

    test_file = "test.pkl"
    if os.path.exists(test_file):
        with open(test_file, "rb") as file:
            test = pickle.load(file)
        if "sentence_transformers_embedings" not in test[0]:
            sentence_transformers_embedings(test)
            with open(test_file, "wb") as file:
                pickle.dump(test, file)
        if "lemmatized_text" not in test[0]:
            lemmatized_text(test)
            with open(test_file, "wb") as file:
                pickle.dump(test, file)
    else:
        test = load("data/rtvslo_test.json")
        emmbedings = sloberta_embedings(test)
        for i, article in enumerate(test):
            article["embeddings"] = emmbedings[i]
        with open(test_file, "wb") as file:
            pickle.dump(test, file)


    n = len(train)
    m = RTVSlo()
    train_data, val_data = train[:(n*4)//5], train[(n*4)//5:]
    m.fit(train_data)
    test_mae(val_data, m)

    p = m.predict(test)
    
    np.savetxt('example1.txt', p, fmt='%f')
