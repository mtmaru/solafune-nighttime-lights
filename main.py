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
        self.nummeshs_mean = None
        self.nummeshs_std = None
        self.year_mean = None
        self.year_std = None

    def fit(self, df):
        df = df.copy()

        df = self._add_features(df)
        df = self._filter(df)

        self.y_mean = df["AverageLandPriceLog"].mean()
        self.y_std = df["AverageLandPriceLog"].std()
        self.meanlight_mean = df["MeanLightLog"].mean()
        self.meanlight_std = df["MeanLightLog"].std()
        self.sumlight_mean = df["SumLightLog"].mean()
        self.sumlight_std = df["SumLightLog"].std()
        self.nummeshs_mean = df["NumMeshsLog"].mean()
        self.nummeshs_std = df["NumMeshsLog"].std()
        self.year_mean = df["Year"].mean()
        self.year_std = df["Year"].std()

        return self

    def transform(self, df):
        df = df.copy()

        df = self._add_features(df)
        df = self._filter(df)
        df = self._transfrom_normalize(df)
        placeids, x, y = self._transfrom_pivot(df)

        return placeids, x, y

    def _add_features(self, df):
        # 土地価格の平均の対数
        df["AverageLandPriceLog"] = np.log(df["AverageLandPrice"] + 1)

        # 夜間光の平均の対数
        df["MeanLightLog"] = np.log(df["MeanLight"] + 1)

        # 夜間光の合計の対数
        df["SumLightLog"] = np.log(df["SumLight"] + 1)

        # メッシュ数の対数
        df["Between2009and2011"] = (2009 <= df["Year"]) & (df["Year"] <= 2011)
        df["NumMeshs"] = np.round(df["SumLight"] / df["MeanLight"])
        nummeshs_mean = df.groupby(["PlaceID", "Between2009and2011"], as_index = False)["NumMeshs"].mean()
        df = df.drop(columns = "NumMeshs").merge(nummeshs_mean, how = "left", on = ["PlaceID", "Between2009and2011"])
        df.drop(columns = ["Between2009and2011"], inplace = True)
        df["NumMeshsLog"] = np.log(df["NumMeshs"] + 1)

        # 年代
        df["Year"] = df["Year"]

        return df

    def _filter(self, df):
        # 22年分揃っているデータに絞る
        num_years = df.groupby("PlaceID").size()
        blacklist = num_years.loc[num_years < 22].index
        df = df.loc[lambda df: ~df["PlaceID"].isin(blacklist), :].copy()

        # メッシュ数が欠損していないデータに絞る
        num_na = df.assign(IsNa = lambda df: df["NumMeshsLog"].isna()).groupby("PlaceID")["IsNa"].sum()
        blacklist = num_na.loc[num_na > 0].index
        df = df.loc[lambda df: ~df["PlaceID"].isin(blacklist), :].copy()

        return df

    def _transfrom_normalize(self, df):
        df["AverageLandPriceLogZ"] = (df["AverageLandPriceLog"] - self.y_mean) / self.y_std
        df["MeanLightLogZ"] = (df["MeanLightLog"] - self.meanlight_mean) / self.meanlight_std
        df["SumLightLogZ"] = (df["SumLightLog"] - self.sumlight_mean) / self.sumlight_std
        df["NumMeshsLogZ"] = (df["NumMeshsLog"] - self.nummeshs_mean) / self.nummeshs_std
        df["YearZ"] = (df["Year"] - self.year_mean) / self.year_std

        return df

    def _transfrom_pivot(self, df):
        # (サンプルサイズ, 年代数) に変形する
        y = df.pivot(index = "PlaceID", columns = "Year", values = "AverageLandPriceLogZ")
        placeids = y.index
        y = y.values

        # (サンプルサイズ, 特徴数, 年代数) に変形する
        x_meanlight = df.pivot(index = "PlaceID", columns = "Year", values = "MeanLightLogZ")
        x_sumlight = df.pivot(index = "PlaceID", columns = "Year", values = "SumLightLogZ")
        x_nummeshs = df.pivot(index = "PlaceID", columns = "Year", values = "NumMeshsLogZ")
        x_year = df.pivot(index = "PlaceID", columns = "Year", values = "YearZ")
        x = np.stack([
            x_meanlight.loc[placeids, :].values,
            x_sumlight.loc[placeids, :].values,
            x_nummeshs.loc[placeids, :].values,
            x_year.loc[placeids, :].values
        ], axis = 1)

        return placeids, x, y

class Estimator:
    def __init__(self, epochs, device):
        self.epochs = epochs
        self.device = device
        self.model = None

    def __getstate__(self):
        state = {
            "epochs": self.epochs,
            "device": self.device,
            "model": self.model.state_dict()
        }

        return state

    def __setstate__(self, state):
        self.epochs = state["epochs"]
        self.device = state["device"]
        self.model = Model().to(state["device"])
        self.model.load_state_dict(state["model"])
    
    def fit(self, x, y, callbacks = []):
        x = torch.from_numpy(x).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)
        dataset = Dataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 128, shuffle = True)
        self.model = Model().to(self.device)
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
        x = torch.from_numpy(x).float().to(self.device)
        self.model.eval()
        y_pred = self.model(x)
        y_pred = y_pred.cpu().detach().numpy()

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

        self.conv_size = 64
        self.hidden_size = 128

        self.conv_1 = nn.Conv2d(in_channels = 4, out_channels = self.conv_size, kernel_size = (3, 1), padding = (1, 0))
        self.conv_dropout_1 = nn.Dropout2d()
        self.conv_2 = nn.Conv2d(in_channels = self.conv_size, out_channels = self.conv_size, kernel_size = (3, 1), padding = (1, 0))
        self.conv_dropout_2 = nn.Dropout2d()
        self.conv_3 = nn.Conv2d(in_channels = self.conv_size, out_channels = self.conv_size, kernel_size = (3, 1), padding = (1, 0))
        self.conv_dropout_3 = nn.Dropout2d()
        self.fc_1 = nn.Linear(self.conv_size * 22, self.hidden_size)
        self.norm_1 = nn.BatchNorm1d(self.hidden_size)
        self.dropout_1 = nn.Dropout()
        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm_2 = nn.BatchNorm1d(self.hidden_size)
        self.dropout_2 = nn.Dropout()
        self.fc_3 = nn.Linear(self.hidden_size, 22)

    def forward(self, x):
        # (batchsize, #features, #years)

        # (batchsize, #features, #years) -> # (batchsize, #features, #years, 1)
        x = x.view(-1, 4, 22, 1)

        # (batchsize, #features, #years, 1) -> (batchsize, conv_size, #years, 1)
        x = self.conv_dropout_1(F.relu(self.conv_1(x)))
        # (batchsize, conv_size, #years, 1) -> (batchsize, conv_size, #years, 1)
        x = self.conv_dropout_2(F.relu(self.conv_2(x)))
        # (batchsize, conv_size, #years, 1) -> (batchsize, conv_size, #years, 1)
        x = self.conv_dropout_3(F.relu(self.conv_3(x)))

        # (batchsize, conv_size, #years, 1) -> (batchsize, conv_size * #years)
        x = torch.flatten(x, start_dim = 1)

        # (batchsize, conv_size * #years) -> (batchsize, hidden_size)
        x = self.dropout_1(F.relu(self.norm_1(self.fc_1(x))))
        # (batchsize, hidden_size) -> (batchsize, hidden_size)
        x = self.dropout_2(F.relu(self.norm_2(self.fc_2(x))))
        # (batchsize, hidden_size) -> (batchsize, #years)
        x = self.fc_3(x)

        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

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
    for i in range(100):
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
        estimator = Estimator(10000, device)
        estimator.fit(train_x, train_y, [print_rmse])

        test_y_pred = estimator.predict(test_x)
        test_rmse = (((test_y - test_y_pred) * preprocessor.y_std) ** 2).mean() ** 0.5
        test_rmse_list.append(test_rmse)

        with open("output/model/model_{:02d}.pickle".format(i), mode = "wb") as fp:
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
    for i in range(100):
        with open("output/model/model_{:02d}.pickle".format(i), mode = "rb") as fp:
            estimator = pickle.load(fp)
        y_pred = y_pred + estimator.predict(x)
    y_pred = y_pred / 100

    y_pred = pd.DataFrame(y_pred, index = placeids, columns = [1992 + i for i in range(22)])
    y_pred = y_pred.stack().rename_axis(["PlaceID", "Year"]).rename("LandPrice").reset_index()
    y_pred["LandPrice"] = np.exp(y_pred["LandPrice"] * preprocessor.y_std + preprocessor.y_mean) - 1
    y_pred.to_csv("output/submission.csv", index = False, header = True)

if __name__ == "__main__":
    args = sys.argv
    if args[1] == "train":
        train()
    elif args[1] == "predict":
        predict()
