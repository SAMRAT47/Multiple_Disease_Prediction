from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import *
from src.logger import logging
from src.utils.main_utils import load_object
from sklearn.impute import SimpleImputer
import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import DiseaseModelEstimator
from dataclasses import dataclass
from src.utils.main_utils import read_yaml_file, save_object, get_schema_file_path, get_target_column, save_numpy_array_data

@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
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
            
            schema_path = get_schema_file_path(self.disease_name)  # ✅ Load schema based on disease
            self._schema_config = read_yaml_file(file_path=schema_path)
            self.target_column = get_target_column(self.disease_name)  # ✅ Dynamically fetch target column  # dynamic schema

        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[DiseaseModelEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = DiseaseModelEstimator(bucket_name=bucket_name,
                                               model_path=model_path,disease_name=self.disease_name)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise MyException(e, sys)
        
    def impute_missing_values(self, heart_df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame instead of ndarray"""
        imputer = SimpleImputer(strategy='mean')
        heart_df = pd.DataFrame(imputer.fit_transform(heart_df), columns=heart_df.columns)
        return heart_df
        
    # def _impute_missing_values_by_class(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
    #     """
    #     Impute missing (NaN) values in numeric columns using the mean of that column grouped by target class.
    #     """
    #     try:
    #         logging.info("Imputing missing values by class (Outcome).")

    #         numeric_cols = diabetes_df.select_dtypes(include=[np.number]).columns.tolist()

    #         if self.target_column not in diabetes_df.columns:
    #             raise Exception(f"{self.target_column} not found in dataframe for class-based imputation.")

    #         for col in numeric_cols:
    #             if col != self.target_column and diabetes_df[col].isnull().sum() > 0:
    #                 diabetes_df[col] = diabetes_df.groupby(self.target_column)[col].transform(lambda x: x.fillna(x.mean()))

    #         return diabetes_df
    #     except Exception as e:
    #         raise MyException(e, sys)
        
    def _impute_outliers_with_iqr(self, heart_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Imputing outliers using IQR method.")
            numeric_cols = heart_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != self.target_column]

            for col in numeric_cols:
                Q1 = heart_df[col].quantile(0.25)
                Q3 = heart_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                heart_df[col] = np.where(heart_df[col] < lower_bound, Q1, heart_df[col])
                heart_df[col] = np.where(heart_df[col] > upper_bound, Q3, heart_df[col])

            return heart_df

        except Exception as e:
            raise MyException(e, sys)

    # def _feature_engineering(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
    #         logging.info("Starting feature engineering...")

    #         if self.disease_name == "diabetes":  # ✅ Apply only for diabetes
    #             diabetes_df['NewBMI'] = pd.cut(
    #                 diabetes_df['BMI'],
    #                 bins=[0, 18.5, 25, 30, 35, np.inf],
    #                 labels=['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
    #             )
    #             diabetes_df['NewInsulinScore'] = diabetes_df['Insulin'].apply(lambda x: "Normal" if 70 <= x <= 130 else "Abnormal")
    #             diabetes_df['NewGlucose'] = pd.cut(
    #                 diabetes_df['Glucose'],
    #                 bins=[0, 70, 99, 125, 200, np.inf],
    #                 labels=['Low', 'Normal', 'Overweight', 'Secret', 'High']
    #             )

    #         # Logging for skipped feature engineering
    #         elif self.disease_name == "heart":
    #             logging.info("Feature engineering skipped for heart disease.")

    #         elif self.disease_name == "kidney":
    #             logging.info("Feature engineering skipped for kidney disease.")

    #         logging.info("Feature engineering done.")
    #         return diabetes_df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(self.target_column, axis=1), test_df[self.target_column]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self.impute_missing_values(x)
            # x = self._impute_missing_values_by_class(x)
            x = self._impute_outliers_with_iqr(x)
            # x = self._feature_engineering(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,disease_name=self.disease_name)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e