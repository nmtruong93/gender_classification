import os

from config.config import settings
from config.log_config import logger
from gender_prediction.data_loader import DataLoader
from gender_prediction.data_processor import DataProcessor
from gender_prediction.trainer import Trainer


class GenderTraining:

    def __init__(self, init_model=False):
        model_folder = os.path.join(settings.BASE_DIR, 'tmp')
        os.makedirs(model_folder, exist_ok=True)
        previous_model_path = os.path.join(model_folder, 'previous_model.txt')
        self.current_model_path = os.path.join(model_folder, 'model.txt')

        if os.path.exists(previous_model_path):
            init_model = True

        trainer = Trainer()
        self.init_model = trainer.load_model(previous_model_path) if init_model else None
        self.cat_cols = []
        self.drop_cols = ['session_id', 'label']

        self.params = {
            'objective': 'binary',
            'boosting': 'gbdt',
            'extra_trees': True,
            'metric': ['binary_logloss', 'auc', 'binary_error'],
            'num_class': 1,
            'learning_rate': 0.01,
            'max_depth': 6,
            'num_leaves': int(2 ** 6 / 2) + 5,
            'min_data_in_leaf': 500,
            'min_gain_to_split': 0.01,
            'feature_fraction': 0.5,
            'num_iterations': 1000,
            'first_metric_only': False,
            'num_threads': 8,
            'seed': 4
        }

        self.best_params = {
            'boosting': 'gbdt',
            'learning_rate': 0.09506659777596786,
            'max_depth': 8,
            'num_leaves': 125,
            'min_gain_to_split': 0.03760794223129103,
            'feature_fraction': 0.6,
            'bagging_fraction': 0.6,
            'bagging_freq': 7,
            'lambda_l1': 0.0001947897509552219,
            'lambda_l2': 0.00019143792676796318,
            'extra_trees': False,
            'max_bin': 64
        }

        self.params.update(self.best_params)

    @staticmethod
    def load_data(train_path, label_path):
        # Load train, valid, test data
        data_loader = DataLoader(train_path=train_path, label_path=label_path)
        raw_df = data_loader.load_data()
        train_df, valid_df, test_df = data_loader.train_valid_test_split(raw_df, by_time=False, test_size=0.1, seed=42)

        return train_df, valid_df, test_df

    def process_data(self, train_df, valid_df, test_df):
        data_processor = DataProcessor()

        df_train = data_processor.get_time_features(train_df)
        df_valid = data_processor.get_time_features(valid_df)
        df_test = data_processor.get_time_features(test_df)

        df_train = data_processor.get_product_features(df_train, is_train_set=True)
        df_valid = data_processor.get_product_features(df_valid)
        df_test = data_processor.get_product_features(df_test)

        X_train, y_train = df_train.drop(['label'], axis=1), df_train['label']
        X_valid, y_valid = df_valid.drop(['label'], axis=1), df_valid['label']
        X_test, y_test = df_test.drop(['label'], axis=1), df_test['label']

        # Cast categorical columns to category type
        time_cat_cols = ['quarter', 'month', 'week', 'day', 'hour', 'minute', 'daysinmonth', 'dayofweek']
        product_cat_cols = [col for col in X_train.columns if 'product_type' in col]
        self.cat_cols = time_cat_cols + product_cat_cols

        for col in self.cat_cols:
            X_train[col] = X_train[col].astype('category')
            X_valid[col] = X_valid[col].astype('category')
            X_test[col] = X_test[col].astype('category')

        return X_train, X_test, X_valid, y_train, y_test, y_valid

    def train(self, X_train, X_test, X_valid, y_train, y_test, y_valid):
        trainer = Trainer(self.params,
                          self.cat_cols,
                          X_train,
                          X_test,
                          X_valid,
                          y_train,
                          y_test,
                          y_valid,
                          init_model=self.init_model
                          )
        # trainer.train()
        trainer.train_optuna(10)
        trainer.save_model(self.current_model_path)
        trainer.plot_feature_importance()

        trainer.evaluate(X_test, y_test)
        trainer.plot_metric()

    def run(self, train_path, label_path):
        raw_data = self.load_data(train_path, label_path)
        processed = self.process_data(*raw_data)
        self.train(*processed)


if __name__ == '__main__':
    train_path = os.path.join(settings.BASE_DIR, 'data', 'trainingData.csv')
    label_path = os.path.join(settings.BASE_DIR, 'data', 'trainingLabels.csv')
    pipeline = GenderTraining()
    pipeline.run(train_path, label_path)
