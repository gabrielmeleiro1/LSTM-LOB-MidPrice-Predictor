import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def load_multiple_files(base_path, start=1, end=9):
    data = []
    for i in range(start, end+1):
        file_path = f"{base_path}_{i}.txt"
        print(f"Loading {file_path}...")
        data.append(np.loadtxt(file_path))
    return np.concatenate(data, axis=1)

def create_dataframes(data):
    df = pd.DataFrame(data[:40, :].T)
    dfAskPrices = df.iloc[:, 0::4]
    dfAskVolumes = df.iloc[:, 1::4]
    dfBidPrices = df.iloc[:, 2::4]
    dfBidVolumes = df.iloc[:, 3::4]
    return dfAskPrices, dfAskVolumes, dfBidPrices, dfBidVolumes

def prepare_data(ask_prices, ask_volumes, bid_prices, bid_volumes, targets, lookback=10, forecast=1):
    df_combined = pd.concat([ask_prices, ask_volumes, bid_prices, bid_volumes], axis=1)
    X, y = [], []
    for i in tqdm(range(len(df_combined) - lookback - forecast + 1), desc="Preparing data"):
        X.append(df_combined.iloc[i:(i+lookback)].values)
        y.append(targets[i+lookback+forecast-1])
    return np.array(X), np.array(y)

def preprocess_data(train_path, test_path, start=1, end=9):
    print("Loading training data...")
    train_data = load_multiple_files(train_path, start, end)
    print("Loading testing data...")
    test_data = load_multiple_files(test_path, start, end)
    
    print("Creating DataFrames...")
    train_ask_prices, train_ask_volumes, train_bid_prices, train_bid_volumes = create_dataframes(train_data)
    test_ask_prices, test_ask_volumes, test_bid_prices, test_bid_volumes = create_dataframes(test_data)
    
    print("Preparing sequences...")
    train_targets = train_data[144:149, :].T
    test_targets = test_data[144:149, :].T
    
    X_train, y_train = prepare_data(train_ask_prices, train_ask_volumes, train_bid_prices, train_bid_volumes, train_targets)
    X_test, y_test = prepare_data(test_ask_prices, test_ask_volumes, test_bid_prices, test_bid_volumes, test_targets)
    
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Testing data shape:", X_test.shape, y_test.shape)
    
    return X_train, y_train, X_test, y_test

def load_and_preprocess_data(base_path):
    train_path = os.path.join(base_path, "train", "Train_Dst_NoAuction_MinMax_CF")
    test_path = os.path.join(base_path, "test", "Test_Dst_NoAuction_MinMax_CF")
    
    return preprocess_data(train_path, test_path)