import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import math
import tqdm

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        np.random.seed(SEED)
        init = np.random.uniform(size= (num_embeddings,embedding_dim))
        self.embeddings.weight.data = torch.FloatTensor(init)

    def forward(self, x):
        x_transpose = x.t()
        x_1 = x_transpose.flatten()
        expend_features = [x_1]
        for i in range(2, self.embedding_dim+1):
            exp_value = torch.pow(x_transpose, i).flatten()
            expend_features.append(exp_value)
        x_result = torch.stack(tuple(expend_features), 1)
        flat_x = x_result
        encoding_indices = self.get_code_indices(flat_x)
        quantized = encoding_indices.view_as(x.T).T

        # embedding loss: move the embeddings towards the encoder's output

        embedding_value = self.quantize(encoding_indices)
        q_latent_loss = 0
        if self.training:
            q_latent_loss = torch.nn.functional.mse_loss(embedding_value, flat_x)
            # commitment loss
            e_latent_loss = torch.nn.functional.mse_loss(x, quantized.detach())
            q_latent_loss = q_latent_loss + 0.25 * e_latent_loss
        # Straight Through Estimator
        quantized = x + (quantized - x).detach().contiguous()
        return quantized, q_latent_loss

    def get_code_indices(self, flat_x):
        distances = (
                torch.sum(flat_x ** 2, dim=1).unsqueeze(1) +
                torch.sum(self.embeddings.weight ** 2, dim=1).unsqueeze(0) -
                2. * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        return encoding_indices
       # flat_x  B * F * E_dim
       #  dimensions = int(flat_x.shape[1]/self.embedding_dim)
       #  extended_embeddings = self.embeddings.weight.repeat(1, dimensions)
       #  distances = torch.abs(flat_x - extended_embeddings)
       #  encoding_indices = distances.reshape(-1, self.embedding_dim)
       #  encoding_indices = torch.argmin(encoding_indices, dim=1)

    def quantize(self, encoding_indices):
        return self.embeddings(encoding_indices)

class GermanNet(nn.Module):
    def __init__(self, D_in, H, D_out, embedding_dim, num_embeddings):
        super(GermanNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim)

    def forward(self, x):
        e, e_q_loss = self.vq_layer(x)
        h1 = self.relu(self.linear1(e))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear2(h2))
        h4 = self.relu(self.linear2(h3))
        h5 = self.relu(self.linear2(h4))
        h6 = self.relu(self.linear2(h5))
        a6 = self.linear3(h6)
        y = self.softmax(a6)
        if not self.training:
            return y
        else:
            return y, e_q_loss

REPEAT_TIME = 3
embedding_dim = 10
num_embeddings = 5
batch_size = 100

def train_model(X_train, y_train, es=True, es_count=10):
    X_train = torch.FloatTensor(X_train)
    y_train = torch.eye(2)[y_train.astype(int)]
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    D_in = X_train.size(1)
    D_out = y_train.size(1)
    epochs = 500
    H = 100

    model = GermanNet(D_in, H, D_out, embedding_dim, num_embeddings)
    model.train()

    lr = 1e-4
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    cur_es_count = 0
    min_loss = 999999
    # Training in batches
    for epoch in range(epochs):
        current_loss = 0
        current_correct = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            output, e_q_loss = model(inputs)
            _, indices = torch.max(output, 1)  # argmax of output [[0.61,0.12]] -> [0]
            preds = torch.eye(2)[indices]
            loss = criterion(output, labels) + e_q_loss
            loss.backward()
            optimizer.step()
            current_loss += loss.item()
            current_correct += (preds.int() == labels.int()).sum() / D_out
        current_loss = current_loss / len(loader)
        current_correct = current_correct / len(torch_dataset)
        if es:
            if min_loss > current_loss:
                min_loss = current_loss
                cur_es_count = 0
            else:
                cur_es_count += 1
                if cur_es_count >= es_count:
                    print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
                    print("early stopped.")
                    return model
        if (epoch % 25 == 0):
            print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
    return model


def get_weights(df, target, show_heatmap=False):
    def heatmap(cor):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
        plt.show()
    cor = df.corr()
    cor_target = abs(cor[target])
    weights = cor_target[:-1]  # removing target WARNING ASSUMES TARGET IS LAST
    weights = weights / np.linalg.norm(weights)
    if show_heatmap:
        heatmap(cor)
    return weights.values

from LowProFool.Adv_origin import lowProFool, deepfool, fgsm

#import Adverse as lpf_multiclass
def gen_adv(model, X_test, y_test, method,alpha, lambda_,bounds, maxiters=1000, weights=None, bar=True):
    results = np.zeros_like(X_test)
    iter = range(X_test.shape[0])
    for i in iter:
        x = X_test[i]
        y = y_test[i]
        x_tensor = torch.FloatTensor(x)
        if method == 'LowProFool':
            orig_pred, adv_pred, x_adv, loop_i = lowProFool(x=x_tensor, model=model, weights=weights, bounds=bounds,
                                                             maxiters=maxiters, alpha=alpha, lambda_=lambda_)
        elif method == 'Deepfool':
            orig_pred, adv_pred, x_adv, loop_i = deepfool(x_tensor, model, maxiters, alpha,
                                                       bounds, weights=[])
        elif method == 'FGSM':
            x_adv = fgsm(x_tensor, model,eps=alpha)
        else:
            raise Exception("Invalid method", method)
        results[i] = x_adv
    return results

def torch_accuracy_score(model, X_test, y_true):
    y_pred = model(X_test)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    return accuracy_score(y_true, y_pred)

def torch_predict(model, data):
    data_tensor = torch.FloatTensor(data)
    model.eval()
    y_pred = model(data_tensor).max(1)[1]
    return y_pred

def evaluate_data(model, y_true, data, pure_pred=None, log=True):
    data_tensor = torch.FloatTensor(data)
    y_pred = model(data_tensor).max(1)[1]
    acc = accuracy_score(y_true, y_pred)
    if pure_pred is not None:
        suc_rate = torch.sum(y_pred != pure_pred)/len(data)
        if log:
            print(f"acc: {acc}, success rate: {suc_rate}")
        return acc, suc_rate
    else:
        return acc

SEED = 0
dataset_list = [
    'CovType',
    'Census-Income',
    'Occupancy_Estimation',
    'Accelerometer',
    'SkinSegmentation',
    'Localization',
    'Higgs',
    'ipums.la.99',
    'connect-4',
    'Adult',
    'letter-recog',
    'magic',
    'GasSensor',
    'sign',
    'satellite',
    'page-blocks',
    'wall-following',
    'waveform-5000',
    'spambase',
    'kr-vs-kp',
    'sick',
    'hypothyroid',
    'cmc',
    'german'
]

#attack param:
maxiters = 50
eps = 0.1
lambda__ = 10 # for LowProFool

from pathlib import Path

out_path = Path('./adv_samples/VQ-VAE/tylor')
for dataset_name in dataset_list:
    print(dataset_name)
    df = pd.read_csv(f'./raw_csv/{dataset_name}.csv', index_col=0)
    df.fillna(0, inplace=True)

    X = df.iloc[:,:-1].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = df.iloc[:,-1].values
    y = LabelEncoder().fit_transform(y)
    if dataset_name == 'satellite':
        idx = np.bitwise_or(y == 0, y ==3)
        X = X[idx]
        y = y[idx]
        y[y == 0] = 0
        y[y == 3] = 1
    else:
        X = X[y < 2]
        y = y[y < 2]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) // 2, shuffle=True, random_state=SEED)

    model = train_model(X_train, y_train)
    model.eval()
    y_pred = model(torch.FloatTensor(X_test))
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
    acc0 = accuracy_score(y_test, y_pred)
    print("Accuracy score on test data", acc0)

    weights = get_weights(df, df.columns[-1])
    bounds = X.min(axis=0), X.max(axis=0)

    if len(X_test) > 300:
        np.random.seed(SEED)
        random_permute = np.random.permutation(len(X_test))
        X_test_sample = X_test[random_permute[:300]]
        y_test_sample = y_test[random_permute[:300]]
    else:
        X_test_sample = X_test
        y_test_sample = y_test

    # Generate adversarial examples
    adv_samples = {}
    for attack_name in ['FGSM', 'LowProFool', 'Deepfool']:
        _p = out_path / f'{dataset_name}_{attack_name.lower()}.npy'
        if _p.exists():
            adv_samples[attack_name] = np.load(_p)
        else:
            print(X_test_sample.shape)
            x_adv = gen_adv(model, X_test_sample, y_test_sample, attack_name, eps, lambda__, bounds, maxiters, weights)
            np.save(_p, x_adv)
            adv_samples[attack_name] = x_adv

    y_test_sample_pred = torch_predict(model, X_test_sample)
    acc0, _ = evaluate_data(model, y_test_sample, X_test_sample, y_test_sample_pred)

    with open(out_path/f'../Tayer.csv', mode='a') as f:
        f.write(','.join([
            dataset_name,
            'Clean',
            'Qlayer',
            str(acc0.item()),
            '0',
        ]))
        f.write('\n')

    for atk_name, atk_raw in adv_samples.items():
        print('----')
        acc1s, sr1s = [],[]
        for r_index in range(REPEAT_TIME):
            acc1, sr1 = evaluate_data(model, y_test_sample, atk_raw, y_test_sample_pred)
            acc1s.append(acc1.item())
            sr1s.append(sr1.item())
        with open(out_path/f'../Taylor.csv', mode='a') as f:
            f.write(','.join([
                dataset_name,
                atk_name,
                'Qlayer-VQ-VAE-Taylor',
                str(np.mean(acc1s)), str(np.mean(sr1s))
            ]))
            f.write('\n')
