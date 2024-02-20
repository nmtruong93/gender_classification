import os
import numpy as np
import lightgbm as lgb

from config.config import settings
from config.log_config import logger


class GenderPrediction:
    def __init__(self, model_path, cat_cols):
        self.model = self.load_model(model_path)
        self.cat_cols = cat_cols

    @staticmethod
    def load_model(path):
        logger.info("Loading model...")
        return lgb.Booster(model_file=path)

    def predict(self, raw_data):
        """
        Predict Gender, 1: Male, 0: Female
        :param raw_data:
        :return:
        """
        y_probs = self.model.predict(raw_data)
        return y_probs


if __name__ == '__main__':
    model_path = os.path.join(settings.BASE_DIR, 'tmp/model.txt')

    predictor = GenderPrediction(model_path, cat_cols=[])
    response_arr = predictor.predict(np.array([1, 2, 3]))
    print(response_arr)

