import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Pytorch
import torch
import torch.nn as nn
import torch.utils.data as Data


class ANNNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ANNNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        # feature number -> 100->2
        h1 = self.relu(self.linear1(x))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear2(h2))
        h4 = self.relu(self.linear2(h3))
        h5 = self.relu(self.linear2(h4))
        h6 = self.relu(self.linear2(h5))
        a6 = self.linear3(h6)
        y = self.softmax(a6)
        return y

class D2Net(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(D2Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.lineard2 = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, D_out)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=-1)

    def get_ew_bins(self, x, levels=5):
        b_min, b_max = x.min(axis=0)[0], x.max(axis=0)[0]
        ew_bins = {}
        columns = ['column' + str(i) for i in range(x.size()[1])]
        for i, col in enumerate(columns):
            ew_bins[col] = torch.tensor(np.linspace(b_min[i], b_max[i], num=levels + 1)[1:-1].tolist())
        return ew_bins

    def get_interp_bins(self, x, levels =5):
        # x
        interp_bins = {}
        columns = ['column' + str(i) for i in range(x.size()[1])]

        values = x.reshape(-1)
        values, _ = torch.sort(values)
        weights = torch.ones_like(values)
        cum_weights = torch.cumsum(weights, 0)
        cum_weight_percents = cum_weights / cum_weights[-1]
        epsilon = 1 / levels
        percents = epsilon + np.arange(0.0, 1.0, epsilon)
        new_bins = np.interp(percents, cum_weight_percents.detach().cpu().numpy(), values.detach().cpu().numpy())[1:]

        for i, col in enumerate(columns):
            interp_bins[col] = torch.tensor(new_bins)
        return interp_bins


    def get_ef_bins(self, x, levels = 5):
        ef_bins = {}
        columns = ['column'+ str(i) for i in range(x.size()[1])]
        percentile_centroids = torch.quantile(x, torch.linspace(0, 100, levels + 1)[1:-1]/100, dim=0).T
        for i, col in enumerate(columns):
            ef_bins[col] = percentile_centroids[i]
        return ef_bins

    def discretize_by_bins(self, x, all_bins: dict):
        X_discre = []
        for i, (col, bins) in enumerate(all_bins.items()):
            _x = x[:, i]
            _x_discre = torch.bucketize(_x, bins)
            X_discre.append(_x_discre.reshape(-1, 1))
        X_discre = torch.hstack(X_discre)
        return torch.tensor(X_discre, dtype=torch.float32, requires_grad=True)

    def descritization(self, x, levels=5):
        bins = self.get_ew_bins(x, levels)
        dis = self.discretize_by_bins(x, bins)
        return dis

    def forward(self, x):
        h1 = self.relu(self.linear1(self.descritization(x, 25)))
        h2 = self.relu(self.linear2(h1))
        h3 = self.relu(self.linear2(h2))
        h4 = self.relu(self.linear2(h3))
        h5 = self.relu(self.linear2(h4))
        h6 = self.relu(self.linear2(h5))
        a6 = self.linear3(h6)
        y = self.softmax(a6)
        return y

def train_model(X_train, y_train, device, modelName='ann', es=True):
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.eye(2)[y_train.astype(int)].to(device)
    torch_dataset = Data.TensorDataset(X_train, y_train)
    loader = Data.DataLoader(
        torch_dataset,
        batch_size=100,
        drop_last=True,
        shuffle=False
    )
    D_in = X_train.size(1)
    D_out = y_train.size(1)
    epochs = 500
    H = 100

    if modelName == 'ann':
        model = ANNNet(D_in, H, D_out).to(device)
    else:
        model = D2Net(D_in, H, D_out).to(device)
    model.train()

    lr = 1e-4
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    es_count = 10
    cur_es_count = 0
    min_loss = 999999
    # Training in batches
    train_accs = []
    train_loss = []
    for epoch in range(epochs):
        current_loss = 0
        current_correct = 0
        for inputs, labels in loader:
            optimizer.zero_grad()
            output = model(inputs)
            _, indices = torch.max(output, 1)  # argmax of output [[0.61,0.12]] -> [0]
            preds = torch.eye(2)[indices].to(device)

            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            current_correct += (preds.int() == labels.int()).sum() / D_out

        current_loss = current_loss / len(loader)
        current_correct = current_correct / len(torch_dataset)
        train_accs.append(current_correct)
        train_loss.append(current_loss)

        if es:
            if min_loss > current_loss:
                min_loss = current_loss
                cur_es_count = 0
            else:
                cur_es_count += 1
                if cur_es_count >= es_count:
                    print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
                    print("early stopped.")
                    return model, epoch
        if (epoch % 50 == 0):
            print("> epoch {:.0f}\tLoss {:.5f}\tAcc {:.5f}".format(epoch, current_loss, current_correct))
    return model, epochs

REPEAT_TIME = 3
num_embeddings = 5
batch_size = 100

def get_weights(df, target):
    cor = df.corr()
    cor_target = abs(cor[target])
    weights = cor_target[:-1]  # removing target WARNING ASSUMES TARGET IS LAST
    weights = weights / np.linalg.norm(weights)
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
    'SkinSegmentation',
    'Localization',
    'Accelerometer',
    'Higgs',
    'ipums.la.99',
    'connect-4',
    'Adult',
    'letter-recog',
    'magic',
    'GasSensor',
    'sign',
    'Occupancy',
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
device = 'cpu'

from pathlib import Path

out_path = Path('./adv_samples/D2-Layer/')
for dataset_name in dataset_list:
    print(dataset_name)
    df = pd.read_csv(f'./raw_csv/{dataset_name}.csv', index_col=0)
    df.fillna(0, inplace=True)

    X = df.iloc[:, :-1].values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    y = df.iloc[:, -1].values
    y = LabelEncoder().fit_transform(y)
    if dataset_name == 'satellite':
        idx = np.bitwise_or(y == 0, y == 3)
        X = X[idx]
        y = y[idx]
        y[y == 0] = 0
        y[y == 3] = 1
    else:
        X = X[y < 2]
        y = y[y < 2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=len(X) // 2, shuffle=True, random_state=SEED)

    d2layer_model, converge_epoch = train_model(X_train, y_train, device, 'd2layer')
    d2layer_model.eval()
    y_pred = d2layer_model(torch.FloatTensor(X_test).to(device))
    y_pred = np.argmax(y_pred.detach().cpu().numpy(), axis=1)
    acc0 = accuracy_score(y_test, y_pred)
    print("Accuracy score on test data", acc0)

    with open(out_path/f'./D2_Clear_epoch.csv', mode='a') as f:
        f.write(','.join([
            dataset_name,
            str(acc0), str(converge_epoch)
        ]))
        f.write('\n')

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
            x_adv = gen_adv(d2layer_model, X_test_sample, y_test_sample, attack_name, eps, lambda__, bounds, maxiters, weights)
            np.save(_p, x_adv)
            adv_samples[attack_name] = x_adv

    y_test_sample_pred = torch_predict(d2layer_model, X_test_sample)
    acc0, _ = evaluate_data(d2layer_model, y_test_sample, X_test_sample, y_test_sample_pred)

    for atk_name, atk_raw in adv_samples.items():
        print('----')
        acc1s, sr1s = [],[]
        for r_index in range(REPEAT_TIME):
            acc1, sr1 = evaluate_data(d2layer_model, y_test_sample, atk_raw, y_test_sample_pred)
            acc1s.append(acc1.item())
            sr1s.append(sr1.item())
        with open(out_path/f'./D2_EW_defense.csv', mode='a') as f:
            f.write(','.join([
                dataset_name,
                atk_name,
                'D2',
                str(np.mean(acc1s)), str(np.mean(sr1s))
            ]))
            f.write('\n')


