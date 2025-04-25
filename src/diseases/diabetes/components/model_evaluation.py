from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from src.entity.s3_estimator import DiseaseModelEstimator
from src.exception import MyException
from src.constants import *
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file, save_object, get_schema_file_path, get_target_column
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import sys


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: Optional[float]
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self, disease_name: str, model_eval_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.disease_name = disease_name
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact

            schema_path = get_schema_file_path(disease_name)
            self._schema_config = read_yaml_file(schema_path)
            self.target_column = get_target_column(disease_name)

        except Exception as e:
            raise MyException(e, sys)

    def get_best_model(self) -> Optional[DiseaseModelEstimator]:
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            estimator = DiseaseModelEstimator(bucket_name=bucket_name, model_path=model_path, disease_name=self.disease_name)

            if estimator.is_model_present(model_path=model_path):
                return estimator
            return None
        except Exception as e:
            raise MyException(e, sys)

    def _replace_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Replacing zeros with NaN.")
            cols = self._schema_config.get("columns_to_replace_zeros", [])
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _impute_outliers_with_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Imputing outliers using IQR.")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            num_cols = [col for col in num_cols if col != self.target_column]

            for col in num_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                df[col] = np.where(df[col] < lower, Q1, df[col])
                df[col] = np.where(df[col] > upper, Q3, df[col])
            return df
        except Exception as e:
            raise MyException(e, sys)

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting feature engineering...")
            if self.disease_name == "diabetes":
                df['NewBMI'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, np.inf],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2'])
                df['NewInsulinScore'] = df['Insulin'].apply(lambda x: "Normal" if 70 <= x <= 130 else "Abnormal")
                df['NewGlucose'] = pd.cut(df['Glucose'], bins=[0, 70, 99, 125, 200, np.inf],
                                          labels=['Low', 'Normal', 'Overweight', 'Secret', 'High'])
            else:
                logging.info(f"No feature engineering applied for {self.disease_name}.")

            return df
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        try:
            df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x = df.drop(self.target_column, axis=1)
            y = df[self.target_column]

            x = self._replace_zeros(x)
            x = self._impute_outliers_with_iqr(x)
            x = self._feature_engineering(x)

            trained_model = load_object(self.model_trainer_artifact.trained_model_file_path)
            trained_model_f1 = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"Trained model F1: {trained_model_f1}")

            best_model = self.get_best_model()
            best_model_f1 = None

            if best_model:
                y_pred_best = best_model.predict(x)
                best_model_f1 = f1_score(y, y_pred_best)
                logging.info(f"Best (prod) model F1: {best_model_f1}")

            is_accepted = (trained_model_f1 > (best_model_f1 or 0))
            diff = trained_model_f1 - (best_model_f1 or 0)

            return EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1,
                best_model_f1_score=best_model_f1,
                is_model_accepted=is_accepted,
                difference=diff
            )

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation...")
            result = self.evaluate_model()

            return ModelEvaluationArtifact(
                is_model_accepted=result.is_model_accepted,
                s3_model_path=self.model_eval_config.s3_model_key_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=result.difference,
                disease_name=self.disease_name
            )
        except Exception as e:
            raise MyException(e, sys)