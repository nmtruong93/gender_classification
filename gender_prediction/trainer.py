import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import optuna
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, f1_score

from config.config import settings
from config.log_config import logger


class Trainer:
    def __init__(self,
                 params=None,
                 cat_cols=None,
                 X_train=None,
                 X_test=None,
                 X_valid=None,
                 y_train=None,
                 y_test=None,
                 y_valid=None,
                 init_model=None
                 ):
        if X_train is not None:
            self.X_train = X_train
            self.y_train = y_train
            self.X_valid = X_valid
            self.y_valid = y_valid
            self.X_test = X_test
            self.y_test = y_test

            self.cat_cols = cat_cols
            self.params = params
            self.evals_result = {}
            self.init_model = init_model

        self.model = None

    def train(self):
        logger.info(f"Start training with params {self.params}")
        train_set = lgb.Dataset(self.X_train,
                                self.y_train,
                                categorical_feature=self.cat_cols)
        valid_set = lgb.Dataset(self.X_valid,
                                self.y_valid,
                                reference=train_set,
                                categorical_feature=self.cat_cols)
        self.model = lgb.train(self.params,
                               train_set=train_set,
                               valid_sets=[train_set, valid_set],
                               valid_names=['train', 'valid'],
                               init_model=self.init_model,
                               keep_training_booster=True,
                               callbacks=[lgb.record_evaluation(self.evals_result),
                                          lgb.log_evaluation(period=100)],
                               # categorical_feature=self.cat_cols,
                               )
        logger.info("Done training...")
        return self.model

    def objective(self, trial):
        self.params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.01, log=True)
        self.params['max_depth'] = trial.suggest_int('max_depth', 5, 12)
        self.params['num_leaves'] = trial.suggest_int('num_leaves', int(2 ** self.params['max_depth'] / 2) - 5,
                                                      int(2 ** self.params['max_depth'] / 2) + 5, step=2)
        self.params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 100, 700, step=100)
        self.params['min_gain_to_split'] = trial.suggest_float('min_gain_to_split', 0.01, 0.1, log=True)
        self.params['feature_fraction'] = trial.suggest_float('feature_fraction', 0.5, 0.8, step=0.1)
        self.params['bagging_fraction'] = trial.suggest_float('bagging_fraction', 0.5, 0.8, step=0.1)
        self.params['bagging_freq'] = trial.suggest_int('bagging_freq', 5, 10)
        self.params['lambda_l1'] = trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True)
        self.params['lambda_l2'] = trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True)
        self.params['max_bin'] = trial.suggest_int('max_bin', 8, 64, step=8)

        self.train()
        trial.set_user_attr('best_model', self.model)

        f1 = f1_score(self.y_valid, self.model.predict(self.X_valid) > 0.5)
        return f1

    def train_optuna(self, n_trials=100):
        logger.info(f"Training optuna with params {self.params}")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        self.model = study.best_trial.user_attrs['best_model']
        logger.info(f"{'*' * 25} Best params: {study.best_params} {'*' * 25}")

        logger.info("Done training...")
        return self.model

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred

    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate the model
        :param X_test: pd.DataFrame, the testing features
        :param y_test: pd.Series, the testing target
        :param threshold: float, the threshold to classify the predicted probabilities
        :return: y_pred_probs: np.array, the predicted probabilities
        """
        logger.info("Start evaluating...")
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.where(y_pred_probs > threshold, 1, 0)

        logger.info(f"Accuracy score: {accuracy_score(y_test, y_pred)}| AUC: {roc_auc_score(y_test, y_pred_probs)}")
        logger.info(f"Classification report:")
        print(classification_report(y_test, y_pred))

        logger.info("Draw confusion matrix...")
        self.draw_confusion_matrix(y_test, y_pred)
        return y_pred_probs

    @staticmethod
    def draw_confusion_matrix(y_test, y_pred):
        """
        Draw the confusion matrix

        :param y_test: pd.Series, the testing target
        :param y_pred: pd.Series, the predicted target
        :return:
        """
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.1, linecolor='black', cbar=False, square=True,
                    annot_kws={'size': 20}, xticklabels=['0', '1'], yticklabels=['0', '1'])
        plt.xlabel('Predicted', fontsize=20)
        plt.ylabel('Actual', fontsize=20)
        plt.show()

    def save_model(self, path):
        logger.info(f"Saving model to path {path}...")
        self.model.save_model(path,
                              num_iteration=self.model.best_iteration)
        logger.info("Done saving model...")

    def load_model(self, path):
        logger.info("Loading model...")
        self.model = lgb.Booster(model_file=path)
        logger.info("Done loading model...")
        return self.model

    def plot_feature_importance(self):
        logger.info("Plotting feature importance...")
        lgb.plot_importance(self.model,
                            importance_type='gain',
                            figsize=(10, 15),
                            height=0.5,
                            max_num_features=50)
        plt.show()
        logger.info("Done plotting feature importance...")

    def plot_metric(self):
        logger.info("Plotting metric...")
        _, ax = plt.subplots(1, 3, figsize=(25, 6))
        lgb.plot_metric(self.evals_result, metric='binary_logloss', title='Log loss', ax=ax[0])
        lgb.plot_metric(self.evals_result, metric='auc', title='AUC', ax=ax[1])
        lgb.plot_metric(self.evals_result, metric='binary_error', title='Error rate', ax=ax[2])
        plt.show()
        logger.info("Done plotting metric...")
