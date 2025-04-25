from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from src.exception import MyException
from src.constants import *
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file, get_schema_file_path, get_target_column
import sys
import pandas as pd
import numpy as np
import os
from typing import Optional
from src.entity.s3_estimator import DiseaseModelEstimator
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder


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
            
            schema_path = get_schema_file_path(self.disease_name)
            self._schema_config = read_yaml_file(file_path=schema_path)
            self.target_column = get_target_column(self.disease_name)

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

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def _apply_column_value_replacements(self, kidney_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Applying manual replacements and schema-based mappings...")

            # Manual replacements for known dirty values
            manual_replacements = {
                'dm': {"\tno": "no", "\tyes": "yes", " yes": "yes", "\t?": np.nan, "??": np.nan},
                'cad': {"\tno": "no", "\t?": np.nan, "??": np.nan},
                'classification': {"ckd\t": "ckd", "notckd": "not ckd", "\t?": np.nan, "??": np.nan}
            }

            for col, replacements in manual_replacements.items():
                if col in kidney_df.columns:
                    kidney_df[col] = kidney_df[col].replace(replacements)

            # General cleanup: replace '\t??' or '??' in all string columns with NaN
            for col in kidney_df.select_dtypes(include='object').columns:
                kidney_df[col] = kidney_df[col].replace({'\t?': np.nan, '??': np.nan})

            # Apply schema-based mapping values
            column_value_mappings = self._schema_config.get("column_value_mappings", {})
            for col, mapping in column_value_mappings.items():
                if col in kidney_df.columns:
                    kidney_df[col] = kidney_df[col].map(mapping)
                    # Convert mapped values to integer if mapping is complete
                    if kidney_df[col].dropna().apply(lambda x: isinstance(x, int)).all():
                        kidney_df[col] = kidney_df[col].astype(int)

            return kidney_df

        except Exception as e:
            raise MyException(e, sys)

    def _rename_columns_as_per_schema(self, kidney_df: pd.DataFrame) -> pd.DataFrame:
        try:
            column_renaming_map = self._schema_config.get("column_renaming_map", {})
            if column_renaming_map:
                logging.info(f"Renaming columns using map: {column_renaming_map}")
                kidney_df = kidney_df.rename(columns=column_renaming_map)
            else:
                logging.info("No column renaming map found in schema.")
            return kidney_df

        except Exception as e:
            logging.exception("Error occurred during column renaming")
            raise MyException(e, sys)

    def _random_sampling_impute(self, kidney_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        try:
            missing_count = kidney_df[feature].isna().sum()
            if missing_count == 0:
                return kidney_df
                
            non_null_values = kidney_df[feature].dropna()
            if missing_count > len(non_null_values):
                # Enable replacement if necessary
                random_sample = non_null_values.sample(missing_count, replace=True, random_state=42)
            else:
                random_sample = non_null_values.sample(missing_count, random_state=42)
            random_sample.index = kidney_df[kidney_df[feature].isna()].index
            kidney_df.loc[kidney_df[feature].isna(), feature] = random_sample
            return kidney_df
        except Exception as e:
            raise MyException(f"Error in random sampling for {feature}: {e}", sys)

    def _mode_impute(self, kidney_df: pd.DataFrame, feature: str) -> pd.DataFrame:
        try:
            if kidney_df[feature].isnull().sum() > 0:
                mode = kidney_df[feature].mode()[0]
                kidney_df[feature] = kidney_df[feature].fillna(mode)
            return kidney_df
        except Exception as e:
            raise MyException(f"Error in mode imputation for {feature}: {e}", sys)

    def _label_encode_columns(self, kidney_df: pd.DataFrame, columns: list) -> pd.DataFrame:
        try:
            for col in columns:
                if col in kidney_df.columns:
                    le = LabelEncoder()
                    kidney_df[col] = le.fit_transform(kidney_df[col].astype(str))  # Handle any potential object type
                else:
                    logging.warning(f"Column {col} not found in dataframe for label encoding")
            return kidney_df
        except Exception as e:
            raise MyException(f"Error in label encoding columns {columns}: {e}", sys)

    def _impute_outliers_with_iqr(self, kidney_df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Imputing outliers using IQR method for numerical features.")
            
            # Use schema to identify numerical columns
            numerical_cols = self._schema_config.get('numerical_columns', [])
            
            # Account for column renaming
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_numerical_cols = [renaming_map.get(col, col) for col in numerical_cols]
            
            # Only use columns that exist in the dataframe
            numerical_cols_to_process = [col for col in renamed_numerical_cols if col in kidney_df.columns]
            numerical_cols_to_process = [col for col in numerical_cols_to_process if col != self.target_column]

            for col in numerical_cols_to_process:
                Q1 = kidney_df[col].quantile(0.25)
                Q3 = kidney_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                kidney_df[col] = np.where(kidney_df[col] < lower_bound, Q1, kidney_df[col])
                kidney_df[col] = np.where(kidney_df[col] > upper_bound, Q3, kidney_df[col])

            return kidney_df

        except Exception as e:
            raise MyException(e, sys)

    def prepare_data_for_evaluation(self, kidney_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare the kidney disease data for model evaluation using the same
        preprocessing steps as in data transformation
        """
        try:
            logging.info("Preparing kidney disease data for evaluation...")
            
            # Clean data using the same methods as in data transformation
            kidney_df = self._apply_column_value_replacements(kidney_df)
            kidney_df = self._rename_columns_as_per_schema(kidney_df)
            
            # Get numerical columns that exist in the dataframe
            numerical_cols = self._schema_config.get("numerical_columns", [])
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_numerical_cols = [renaming_map.get(col, col) for col in numerical_cols]
            existing_numerical_cols = [col for col in renamed_numerical_cols if col in kidney_df.columns]
            
            # Apply imputation for numerical columns
            for col in existing_numerical_cols:
                kidney_df = self._random_sampling_impute(kidney_df, col)
            
            # Apply imputation for specific domain columns if they exist
            for col in ["red_blood_cells", "pus_cell"]:  # Using renamed columns here
                if col in kidney_df.columns:
                    kidney_df = self._random_sampling_impute(kidney_df, col)
            
            # Get categorical columns that exist in the dataframe
            categorical_cols = self._schema_config.get("categorical_columns", [])
            renamed_categorical_cols = [renaming_map.get(col, col) for col in categorical_cols]
            existing_categorical_cols = [col for col in renamed_categorical_cols if col in kidney_df.columns]
            
            # Apply mode imputation for categorical columns
            for col in existing_categorical_cols:
                kidney_df = self._mode_impute(kidney_df, col)
            
            # Apply label encoding for categorical columns
            kidney_df = self._label_encode_columns(kidney_df, existing_categorical_cols)
            
            # Handle outliers in numerical features
            kidney_df = self._impute_outliers_with_iqr(kidney_df)
            
            # Verify target column exists and handle column renaming if needed
            if self.target_column not in kidney_df.columns:
                # Check if the target column needs to be renamed
                renamed_target = renaming_map.get(self.target_column, self.target_column)
                if renamed_target in kidney_df.columns:
                    self.target_column = renamed_target
                else:
                    raise ValueError(f"Target column '{self.target_column}' or its renamed version '{renamed_target}' not found in dataframe. Available columns: {kidney_df.columns.tolist()}")
            
            logging.info(f"Data preparation completed with final columns: {kidney_df.columns.tolist()}")
            return kidney_df
            
        except Exception as e:
            raise MyException(e, sys) from e

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Load test data
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)
            logging.info(f"Test data loaded with columns: {test_df.columns.tolist()}")
            
            # Prepare data using the same logic as data transformation
            test_df = self.prepare_data_for_evaluation(test_df)
            
            # Split features and target
            if self.target_column not in test_df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in preprocessed dataframe")
                
            x, y = test_df.drop(self.target_column, axis=1), test_df[self.target_column]
            logging.info(f"Features shape: {x.shape}, Target shape: {y.shape}")

            # Load trained model
            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded successfully.")
            
            # Get F1 score from training artifact
            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score
            logging.info(f"F1_Score for trained model: {trained_model_f1_score}")

            # Get best model from production if available
            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info("Production model found. Computing F1 score for comparison...")
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            else:
                logging.info("No production model found. New model will be considered for deployment.")
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Model evaluation result: {result}")
            return result

        except Exception as e:
            logging.exception("Exception occurred during model evaluation")
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            logging.info("=" * 60)
            logging.info(f"Initializing Model Evaluation for {self.disease_name} disease.")
            
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,disease_name=self.disease_name)

            logging.info(f"Model evaluation completed with artifact: {model_evaluation_artifact}")
            logging.info("=" * 60)
            return model_evaluation_artifact
            
        except Exception as e:
            logging.exception("Exception occurred in model evaluation")
            raise MyException(e, sys) from e