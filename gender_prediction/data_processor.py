import os
import pickle
import numpy as np
from tqdm import tqdm

from config.config import settings
from config.log_config import logger
tqdm.pandas()


class DataProcessor:
    def __init__(self):
        self.drop_cols = ['session_id', 'start_time', 'end_time', 'product_viewed', 'product_list',
                          'product_type_0', 'product_type_1', 'product_type_2']
        self.cat_cols = []

        self.product_type_0 = None
        self.product_type_1 = None
        self.product_type_2 = None

    @staticmethod
    def get_cycled_feature_value_sin(col, max_value, epsilon=0.000001):
        value_scaled = (col + epsilon) / max_value
        value_sin = np.sin(2 * np.pi * value_scaled)
        return value_sin

    @staticmethod
    def get_cycled_feature_value_cos(col, max_value, epsilon=0.000001):
        value_scaled = (col + epsilon) / max_value
        value_cos = np.cos(2 * np.pi * value_scaled)
        return value_cos

    def get_time_features(self, df, time_col='start_time'):
        """
        Get time features
        :param df: pd.DataFrame
        :param time_col: str, default='start_time'
        :return: pd.DataFrame
        """
        logger.info("Get time features...")
        df['quarter'] = df[time_col].dt.quarter
        df['month'] = df[time_col].dt.month
        df['week'] = df[time_col].dt.week
        df['day'] = df[time_col].dt.day
        df['hour'] = df[time_col].dt.hour
        df['minute'] = df[time_col].dt.minute
        df['daysinmonth'] = df[time_col].dt.daysinmonth
        df['dayofweek'] = df[time_col].dt.dayofweek
        df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds()

        # Calculate cyclic features
        df['hour_sin'] = self.get_cycled_feature_value_sin(df['hour'], 24)
        df['hour_cos'] = self.get_cycled_feature_value_cos(df['hour'], 24)
        df['dayofweek_sin'] = self.get_cycled_feature_value_sin(df['dayofweek'], 7)
        df['dayofweek_cos'] = self.get_cycled_feature_value_cos(df['dayofweek'], 7)
        df['day_sin'] = self.get_cycled_feature_value_sin(df['day'], df['daysinmonth'])
        df['day_cos'] = self.get_cycled_feature_value_cos(df['day'], df['daysinmonth'])
        df['month_sin'] = self.get_cycled_feature_value_sin(df['month'], 12)
        df['month_cos'] = self.get_cycled_feature_value_cos(df['month'], 12)

        # Normalize cyclic features
        df['hour_sin_norm'] = df['hour_sin'] / df['hour_sin'].max()
        df['hour_cos_norm'] = df['hour_cos'] / df['hour_cos'].max()
        df['dayofweek_sin_norm'] = df['dayofweek_sin'] / df['dayofweek_sin'].max()
        df['dayofweek_cos_norm'] = df['dayofweek_cos'] / df['dayofweek_cos'].max()
        df['day_sin_norm'] = df['day_sin'] / df['day_sin'].max()
        df['day_cos_norm'] = df['day_cos'] / df['day_cos'].max()
        df['month_sin_norm'] = df['month_sin'] / df['month_sin'].max()
        df['month_cos_norm'] = df['month_cos'] / df['month_cos'].max()

        return df

    @staticmethod
    def get_product_type(x, level):
        """
        Get product type at level 0, 1, 2, 3
        :param x: list, product list. Example: ['a/b/c', 'a/b/d']
        :return: list, product type at level 0, 1, 2, 3. Example: ['a', 'b', 'c', 'd']
        """
        return [y.split('/')[level] for y in x]

    def get_product_features(self, df, is_train_set=False):
        """
        Get product features
        :param df: pd.DataFrame
        :param is_train_set: bool, default=False. If True, process for train set
        :return: pd.DataFrame
        """
        logger.info("Get session features...")
        df['product_list'] = df['product_viewed'].apply(lambda x: x.split(';'))
        df['num_products'] = df['product_list'].apply(lambda x: len(x))
        df['num_unique_products'] = df['product_list'].apply(lambda x: len(set(x)))

        # Get all level product categories
        df['product_type_0'] = df['product_list'].apply(lambda x: self.get_product_type(x, 0))
        df['product_type_1'] = df['product_list'].apply(lambda x: self.get_product_type(x, 1))
        df['product_type_2'] = df['product_list'].apply(lambda x: self.get_product_type(x, 2))
        # df['product_type_3'] = df['product_list'].apply(lambda x: self.get_product_type(x, 3))

        # List of all product categories
        if is_train_set:
            self.product_type_0 = list(set(df['product_type_0'].sum()))
            self.product_type_1 = list(set(df['product_type_1'].sum()))
            self.product_type_2 = list(set(df['product_type_2'].sum()))
            # product_type_3 = list(set(df['product_type_3'].sum()))

            # Save product type to pickle file for encode valid and test data
            with open(os.path.join(settings.BASE_DIR, 'product_type_0.pkl'), 'wb') as f:
                pickle.dump(self.product_type_0, f)
            with open(os.path.join(settings.BASE_DIR, 'product_type_1.pkl'), 'wb') as f:
                pickle.dump(self.product_type_1, f)
            with open(os.path.join(settings.BASE_DIR, 'product_type_2.pkl'), 'wb') as f:
                pickle.dump(self.product_type_2, f)

        # One hot encoding for product type
        for p in self.product_type_0:
            df[f'product_type_0_{p}'] = df['product_type_0'].apply(lambda x: p in x).astype(int)
        for p in self.product_type_1:
            df[f'product_type_1_{p}'] = df['product_type_1'].apply(lambda x: p in x).astype(int)
        for p in self.product_type_2:
            df[f'product_type_2_{p}'] = df['product_type_2'].apply(lambda x: p in x).astype(int)

        return df.drop(self.drop_cols, axis=1)
