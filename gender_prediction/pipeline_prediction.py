import os

from config.config import settings
from config.log_config import logger
from gender_prediction.data_processor import DataProcessor
from gender_prediction.data_loader import DataLoader
from gender_prediction.trainer import Trainer


class GenderPrediction:
    def __init__(self, model_path):
        self.trainer = Trainer()
        self.trainer.load_model(model_path)
        self.cat_cols = []

    def process_data(self, raw_data):
        """
        Process raw data
        :param raw_data: pd.DataFrame
        :return: pd.DataFrame
        """
        data_processor = DataProcessor()
        df = data_processor.get_time_features(raw_data)
        df = data_processor.get_product_features(df)
        X = df.drop(['label'], axis=1)
        y = df['label']

        # Cast categorical columns to category type
        time_cat_cols = ['quarter', 'month', 'week', 'day', 'hour', 'minute', 'daysinmonth', 'dayofweek']
        product_cat_cols = [col for col in X.columns if 'product_type' in col]
        self.cat_cols = time_cat_cols + product_cat_cols

        for col in self.cat_cols:
            X[col] = X[col].astype('category')
        return X, y

    def predict(self, raw_data):
        """
        Predict Gender, 1: Male, 0: Female
        :param raw_data:
        :return:
        """
        X, y = self.process_data(raw_data)
        y_probs = self.trainer.predict(X)
        self.trainer.evaluate(X, y)
        return y_probs


if __name__ == '__main__':
    train_path = os.path.join(settings.BASE_DIR, 'data', 'trainingData.csv')
    label_path = os.path.join(settings.BASE_DIR, 'data', 'trainingLabels.csv')
    data_loader = DataLoader(train_path=train_path, label_path=label_path)
    raw_df = data_loader.load_data(nrows=None)

    model_path = os.path.join(settings.BASE_DIR, 'tmp/model.txt')
    predictor = GenderPrediction(model_path)
    response_arr = predictor.predict(raw_df)
    print(response_arr)

