import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_object, get_schema_file_path, get_target_column, save_numpy_array_data
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact
from src.constants import *

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.disease_name = self.data_transformation_config.disease_name

            schema_path = get_schema_file_path(self.disease_name)
            self._schema_config = read_yaml_file(file_path=schema_path)
            self.target_column = get_target_column(self.disease_name)

            self.heart_df = None  # For column existence checks

        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def _ensure_dir(file_path: str):
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def impute_missing_values(self, heart_df: pd.DataFrame) -> pd.DataFrame:
        """Returns a DataFrame instead of ndarray"""
        imputer = SimpleImputer(strategy='mean')
        heart_df = pd.DataFrame(imputer.fit_transform(heart_df), columns=heart_df.columns)
        return heart_df

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


    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")
        try:
            normalizer = Normalizer()
            standard_scaler = StandardScaler()
            numerical_imputer = SimpleImputer(strategy='mean')  # For imputing numerical values
            categorical_imputer = SimpleImputer(strategy='most_frequent')  # For imputing categorical values

            # Fetch columns from schema, excluding the target column
            normalization_columns = [col for col in self._schema_config['normalization_columns'] if col != self.target_column]
            standard_scaler_columns = [col for col in self._schema_config['standard_scaler_columns'] if col != self.target_column]
            categorical_columns = [col for col in self._schema_config['categorical_columns'] if col != self.target_column]

            logging.info("Columns loaded from schema")

            if self.heart_df is None:
                raise ValueError("heart_df not set before calling get_data_transformer_object")

            missing_columns = [col for col in normalization_columns + standard_scaler_columns + categorical_columns if col not in self.heart_df.columns]
            if missing_columns:
                logging.error(f"Missing columns in dataframe: {missing_columns}")
                raise ValueError(f"Missing columns: {missing_columns}")

            # Adding imputers to the pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("NumericalImputer", numerical_imputer, normalization_columns + standard_scaler_columns),
                    ("CategoricalImputer", categorical_imputer, categorical_columns),
                    ("Normalizer", normalizer, normalization_columns),
                    ("StandardScaler", standard_scaler, standard_scaler_columns)
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])

            logging.info("Final Pipeline Ready!!")

            # Saving the preprocessor object
            preprocessing_object_file_name = DISEASES[self.disease_name]['preprocessing_object_file_name']
            preprocessing_object_path = os.path.join(
                self.data_transformation_config.data_transformation_dir,
                "transformed_object",
                preprocessing_object_file_name
            )

            os.makedirs(os.path.dirname(preprocessing_object_path), exist_ok=True)
            save_object(preprocessing_object_path, preprocessor)

            return final_pipeline,preprocessing_object_path

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object")
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

             # Impute missing values separately
            train_df= self.impute_missing_values(train_df)
            test_df= self.impute_missing_values(test_df)

            train_df = self._impute_outliers_with_iqr(train_df)
            test_df = self._impute_outliers_with_iqr(test_df)

            input_feature_train_df = train_df.drop(columns=[self.target_column], axis=1)
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column], axis=1)
            target_feature_test_df = test_df[self.target_column]

            # Set sample df for column validation in transformer
            self.heart_df = input_feature_train_df

            preprocessor,preprocessing_object_path = self.get_data_transformer_object()
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("feature-target concatenation done for train-test df.")

            self._ensure_dir(self.data_transformation_config.transformed_train_file_path)
            self._ensure_dir(self.data_transformation_config.transformed_test_file_path)
            self._ensure_dir(self.data_transformation_config.transformed_object_file_path)

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_object_path,
                disease_name=self.disease_name
                        )

        except Exception as e:
            raise MyException(e, sys) from e