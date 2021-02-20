import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

torch.manual_seed(0)

class Preprocessor:
    def __init__(self):
        self.y_mean = None
        self.y_std = None
        self.meanlight_mean = None
        self.meanlight_std = None
        self.sumlight_mean = None
        self.sumlight_std = None

    def fit(self, df):
        df = df.copy()

        # log(AverageLandPrice + 1) の平均と標準偏差を求める
        df["AverageLandPrice"] = np.log(df["AverageLandPrice"] + 1)
        self.y_mean = df["AverageLandPrice"].mean()
        self.y_std = df["AverageLandPrice"].std()

        # log(MeanLight + 1) の平均と標準偏差を求める
        df["MeanLight"] = np.log(df["MeanLight"] + 1)
        self.meanlight_mean = df["MeanLight"].mean()
        self.meanlight_std = df["MeanLight"].std()

        # log(SumLight + 1) の平均と標準偏差を求める
        df["SumLight"] = np.log(df["SumLight"] + 1)
        self.sumlight_mean = df["SumLight"].mean()
        self.sumlight_std = df["SumLight"].std()

        # log(SumLight / MeanLight + 1) の平均と標準偏差を求める
        df["NumMeshs"] = np.round(df["SumLight"] / df["MeanLight"])
        df["NumMeshs"].fillna(0.0, inplace = True)
        df["NumMeshs"] = np.log(df["NumMeshs"] + 1)
        self.nummeshs_mean = df["NumMeshs"].mean()
        self.nummeshs_std = df["NumMeshs"].std()

        return self

    def transform(self, df):
        df = df.copy()

        # log(AverageLandPrice + 1) を標準化する
        df["AverageLandPrice"] = np.log(df["AverageLandPrice"] + 1)
        df["AverageLandPrice"] = (df["AverageLandPrice"] - self.y_mean) / self.y_std

        # log(SumLight / MeanLight + 1) を標準化する
        df["NumMeshs"] = np.round(df["SumLight"] / df["MeanLight"])
        df["NumMeshs"].fillna(0.0, inplace = True)
        df["NumMeshs"] = np.log(df["NumMeshs"] + 1)
        df["NumMeshs"] = (df["NumMeshs"] - self.nummeshs_mean) / self.nummeshs_std

        # log(MeanLight + 1) を標準化する
        df["MeanLight"] = np.log(df["MeanLight"] + 1)
        df["MeanLight"] = (df["MeanLight"] - self.meanlight_mean) / self.meanlight_std

        # log(SumLight + 1) を標準化する
        df["SumLight"] = np.log(df["SumLight"] + 1)
        df["SumLight"] = (df["SumLight"] - self.sumlight_mean) / self.sumlight_std

        # (サンプルサイズ, 年代数) に変形する
        y = df.pivot(index = "PlaceID", columns = "Year", values = "AverageLandPrice").dropna()
        placeids = y.index
        y = y.values

        # (サンプルサイズ, 特徴数, 年代数) に変形する
        x_meanlight = df.pivot(index = "PlaceID", columns = "Year", values = "MeanLight").dropna()
        x_sumlight = df.pivot(index = "PlaceID", columns = "Year", values = "SumLight").dropna()
        x_nummeshs = df.pivot(index = "PlaceID", columns = "Year", values = "NumMeshs").dropna()
        x = np.stack([x_meanlight.loc[placeids, :].values, x_sumlight.loc[placeids, :].values, x_nummeshs.loc[placeids, :].values], axis = 1)

        return placeids, x, y

class Estimator:
    def __init__(self):
        self.epochs = 10000
        self.model = None

    def __getstate__(self):
        state = {
            "epochs": self.epochs,
            "model": self.model.state_dict()
        }

        return state

    def __setstate__(self, state):
        self.epochs = state["epochs"]
        self.model = Model()
        self.model.load_state_dict(state["model"])
    
    def fit(self, x, y, callbacks = []):
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        dataset = Dataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)
        self.model = Model()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())
        for epoch in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                batch_y_pred = self.model(batch_x)
                loss = criterion(batch_y_pred, batch_y)
                loss.backward()
                optimizer.step()
            if any([callback(epoch, self, locals()) for callback in callbacks]):
                break

        return self

    def predict(self, x):
        x = torch.from_numpy(x).float()
        self.model.eval()
        y_pred = self.model(x)
        y_pred = y_pred.detach().numpy()

        return y_pred

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index, :, :], self.y[index, :]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv_1 = nn.Conv2d(3, 4, (3, 1))
        self.conv_dropout_1 = nn.Dropout2d()
        self.conv_2 = nn.Conv2d(4, 8, (3, 1))
        self.conv_dropout_2 = nn.Dropout2d()
        self.conv_3 = nn.Conv2d(8, 16, (3, 1))
        self.conv_dropout_3 = nn.Dropout2d()
        self.fc_1 = nn.Linear(256, 128)
        self.norm_1 = nn.BatchNorm1d(128)
        self.dropout_1 = nn.Dropout()
        self.fc_2 = nn.Linear(128, 128)
        self.norm_2 = nn.BatchNorm1d(128)
        self.dropout_2 = nn.Dropout()
        self.fc_3 = nn.Linear(128, 22)

    def forward(self, x):
        # (batchsize, #features = 3, #years = 22)

        # (batchsize, 3, 22) -> # (batchsize, 3, 22, 1)
        x = x.view(-1, 3, 22, 1)

        # (batchsize, 3, 22, 1) -> (batchsize, 4, 20, 1)
        x = self.conv_dropout_1(F.relu(self.conv_1(x)))
        # (batchsize, 4, 20, 1) -> (batchsize, 8, 18, 1)
        x = self.conv_dropout_2(F.relu(self.conv_2(x)))
        # (batchsize, 8, 18, 1) -> (batchsize, 16, 16, 1)
        x = self.conv_dropout_3(F.relu(self.conv_3(x)))

        # (batchsize, 16, 16, 1) -> (batchsize, 256)
        x = torch.flatten(x, start_dim = 1)

        # (batchsize, 256) -> (batchsize, 128)
        x = self.dropout_1(F.relu(self.norm_1(self.fc_1(x))))
        # (batchsize, 128) -> (batchsize, 128)
        x = self.dropout_2(F.relu(self.norm_2(self.fc_2(x))))
        # (batchsize, 128) -> (batchsize, 22)
        x = self.fc_3(x)

        return x

def train():
    train = pd.read_csv(
        "input/TrainDataSet.csv",
        dtype = {
            "PlaceID": int,
            "Year": int,
            "MeanLight": float,
            "SumLight": float,
            "AverageLandPrice": float
        }
    )

    preprocessor = Preprocessor()
    preprocessor = preprocessor.fit(train)
    with open("output/model/preprocessor.pickle", mode = "wb") as fp:
        pickle.dump(preprocessor, fp)
    placeids, x, y = preprocessor.transform(train)

    test_rmse_list = []
    for i in range(10):
        print("======== Estimator {} ========".format(i))

        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2, random_state = i)
        def print_rmse(epoch, estimator, locals):
            if (epoch + 1) % 100 == 0:
                train_y_pred = estimator.predict(train_x)
                train_rmse = (((train_y - train_y_pred) * preprocessor.y_std) ** 2).mean() ** 0.5
                test_y_pred = estimator.predict(test_x)
                test_rmse = (((test_y - test_y_pred) * preprocessor.y_std) ** 2).mean() ** 0.5
                print("Epoch: {:4d}, RMSE (Training): {:.4f}, RMSE (Test): {:.4f}".format(epoch, train_rmse, test_rmse))
            return False
        estimator = Estimator()
        estimator.fit(train_x, train_y, [print_rmse])

        test_y_pred = estimator.predict(test_x)
        test_rmse = (((test_y - test_y_pred) * preprocessor.y_std) ** 2).mean() ** 0.5
        test_rmse_list.append(test_rmse)

        with open("output/model/model_{}.pickle".format(i), mode = "wb") as fp:
            pickle.dump(estimator, fp)

    print("RMSE: mean {:.4f} std {:.4f}".format(
        np.mean(test_rmse_list),
        np.std(test_rmse_list)
    ))

def predict():
    test = pd.read_csv(
        "input/EvaluationData.csv",
        dtype = {
            "PlaceID": int,
            "Year": int,
            "MeanLight": float,
            "SumLight": float
        }
    )
    test["AverageLandPrice"] = 0  # ダミー

    with open("output/model/preprocessor.pickle", mode = "rb") as fp:
        preprocessor = pickle.load(fp)
    placeids, x, y = preprocessor.transform(test)

    y_pred = np.zeros(y.shape)
    for i in range(10):
        with open("output/model/model_{}.pickle".format(i), mode = "rb") as fp:
            estimator = pickle.load(fp)
        y_pred = y_pred + estimator.predict(x)
    y_pred = y_pred / 10

    y_pred = pd.DataFrame(y_pred, index = placeids, columns = [1992 + i for i in range(22)])
    y_pred = y_pred.stack().rename_axis(["PlaceID", "Year"]).rename("LandPrice").reset_index()
    y_pred["LandPrice"] = np.exp(y_pred["LandPrice"] * preprocessor.y_std + preprocessor.y_mean) - 1
    y_pred.to_csv("output/submittion.csv", index = False, header = True)

if __name__ == "__main__":
    args = sys.argv
    if args[1] == "train":
        train()
    elif args[1] == "predict":
        predict()
