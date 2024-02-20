import os
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from config.config import settings
from config.log_config import logger


class DataLoader:
    def __init__(self, train_path, label_path):
        self.train_path = train_path
        self.label_path = label_path

    def load_data(self, nrows=None):
        """
        Load data from path
        :param nrows: int, number of rows will be loaded
        :return: pd.DataFrame
        """
        logger.info(f"Load data from {self.train_path} and {self.label_path}...")
        train_raw = pd.read_csv(self.train_path,
                                header=None,
                                names=['session_id', 'start_time', 'end_time', 'product_viewed'],
                                parse_dates=['start_time', 'end_time'],
                                nrows=nrows,
                                )

        label_raw = pd.read_csv(self.label_path,
                                header=None,
                                names=['gender'],
                                nrows=nrows
                                )
        label_raw.loc[label_raw.gender == 'male', 'label'] = 1
        label_raw.loc[label_raw.gender == 'female', 'label'] = 0
        logger.info(f"Gender ratio: {label_raw.gender.value_counts(normalize=True).to_dict()}")

        df = pd.concat([train_raw, label_raw], axis=1)

        return df.drop(columns=['gender'])

    @staticmethod
    def train_valid_test_split(df, test_size=0.15, seed=42, by_time=True, num_days=3):
        """
        Split data into train, test, validation
        :param df:
        :param test_size: float, test size
        :param seed: int, random seed
        :param by_time: bool, if True, split data by time, else split data randomly
        :param num_days: int, number of days for test and validation set
        :return:
        """
        logger.info(f"Splitting data into train, test, validation...")
        if by_time:
            max_date = df['start_time'].max()
            test_date = max_date - timedelta(days=num_days)
            valid_date = test_date - timedelta(days=num_days)

            df_train = df[df['start_time'] < valid_date]
            df_valid = df[(df['start_time'] >= valid_date) & (df['start_time'] < test_date)]
            df_test = df[df['start_time'] >= test_date]
            logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}, Validation shape: {df_valid.shape}")
            return df_train, df_valid, df_test

        test_size = int(df.shape[0] * test_size)
        df_train, df_test = train_test_split(df,
                                             test_size=test_size,
                                             shuffle=True,
                                             stratify=df['label'],
                                             random_state=seed,
                                             )
        df_train, df_valid = train_test_split(df_train,
                                              test_size=test_size,
                                              shuffle=True,
                                              stratify=df_train['label'],
                                              random_state=seed
                                              )

        logger.info(f"Train shape: {df_train.shape}, Test shape: {df_test.shape}, Validation shape: {df_valid.shape}")
        logger.info(f"Train label ratio: {df_train.label.value_counts(normalize=True).to_dict()}, "
                    f"validation label ratio: {df_valid.label.value_counts(normalize=True).to_dict()}, "
                    f"test label ratio: {df_test.label.value_counts(normalize=True).to_dict()}")
        return df_train, df_valid, df_test
