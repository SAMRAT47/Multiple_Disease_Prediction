import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
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
            self.disease_name = self.data_transformation_config.disease_name  # ✅ Consistent naming

            schema_path = get_schema_file_path(self.disease_name)  # ✅ Load schema based on disease
            self._schema_config = read_yaml_file(file_path=schema_path)
            self.target_column = get_target_column(self.disease_name)  # ✅ Dynamically fetch target column

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

    def _replace_zeros(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
        """Replace zero values in specified columns with NaN for later imputation."""
        try:
            logging.info("Replacing zero values with NaN.")
            columns = self._schema_config['columns_to_replace_zeros']
            for col in columns:
                if col in diabetes_df.columns:
                    diabetes_df[col] = diabetes_df[col].replace(0, np.nan)
            return diabetes_df
        except Exception as e:
            raise MyException(e, sys)

    def _impute_missing_values_by_class(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing (NaN) values in numeric columns using the mean of that column grouped by target class.
        """
        try:
            logging.info("Imputing missing values by class (Outcome).")

            numeric_cols = diabetes_df.select_dtypes(include=[np.number]).columns.tolist()

            if self.target_column not in diabetes_df.columns:
                raise Exception(f"{self.target_column} not found in dataframe for class-based imputation.")

            for col in numeric_cols:
                if col != self.target_column and diabetes_df[col].isnull().sum() > 0:
                    diabetes_df[col] = diabetes_df.groupby(self.target_column)[col].transform(lambda x: x.fillna(x.mean()))

            return diabetes_df
        except Exception as e:
            raise MyException(e, sys)

    def _impute_outliers_with_iqr(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers in numeric columns using IQR method:
        - Values below Q1 - 1.5*IQR will be replaced with Q1
        - Values above Q3 + 1.5*IQR will be replaced with Q3
        """
        try:
            logging.info("Imputing outliers using IQR method.")
            numeric_cols = diabetes_df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != self.target_column]

            for col in numeric_cols:
                Q1 = diabetes_df[col].quantile(0.25)
                Q3 = diabetes_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                diabetes_df[col] = np.where(diabetes_df[col] < lower_bound, Q1, diabetes_df[col])
                diabetes_df[col] = np.where(diabetes_df[col] > upper_bound, Q3, diabetes_df[col])

            return diabetes_df

        except Exception as e:
            raise MyException(e, sys)

    def _feature_engineering(self, diabetes_df: pd.DataFrame) -> pd.DataFrame:
            logging.info("Starting feature engineering...")

            if self.disease_name == "diabetes":  # ✅ Apply only for diabetes
                diabetes_df['NewBMI'] = pd.cut(
                    diabetes_df['BMI'],
                    bins=[0, 18.5, 25, 30, 35, np.inf],
                    labels=['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
                )
                diabetes_df['NewInsulinScore'] = diabetes_df['Insulin'].apply(lambda x: "Normal" if 70 <= x <= 130 else "Abnormal")
                diabetes_df['NewGlucose'] = pd.cut(
                    diabetes_df['Glucose'],
                    bins=[0, 70, 99, 125, 200, np.inf],
                    labels=['Low', 'Normal', 'Overweight', 'Secret', 'High']
                )

            # Logging for skipped feature engineering
            elif self.disease_name == "heart":
                logging.info("Feature engineering skipped for heart disease.")

            elif self.disease_name == "kidney":
                logging.info("Feature engineering skipped for kidney disease.")

            logging.info("Feature engineering done.")
            return diabetes_df

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            robust_scaler = RobustScaler()
            standard_scaler = StandardScaler()
            one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

            logging.info("Transformers Initialized: RobustScaler, StandardScaler, OneHotEncoder")

            # Load schema configurations dynamically based on the disease name
            robust_scaler_columns = self._schema_config['robust_scaler_columns']
            standard_scaler_columns = self._schema_config['standard_scaler_columns']
            ohe_columns = self._schema_config['columns_to_apply_one_hot_encoding']

            logging.info("Columns loaded from schema")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("RobustScaler", robust_scaler, robust_scaler_columns),
                    ("StandardScaler", standard_scaler, standard_scaler_columns),
                    ("OneHotEncoder", one_hot_encoder, ohe_columns)
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")

            # Save preprocessing object dynamically based on disease type
            preprocessing_object_file_name = DISEASES[self.disease_name]['preprocessing_object_file_name']

            # Construct the full path using the configured directory
            preprocessing_object_path = os.path.join(
                self.data_transformation_config.data_transformation_dir,
                "transformed_object",
                preprocessing_object_file_name
            )

            os.makedirs(os.path.dirname(preprocessing_object_path), exist_ok=True)
            # Now save to the full path
            save_object(preprocessing_object_path, preprocessor)

            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object")
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Replace zeros
            train_df = self._replace_zeros(train_df)
            test_df = self._replace_zeros(test_df)

            # Impute missing values with class-wise mean
            train_df = self._impute_missing_values_by_class(train_df)
            test_df = self._impute_missing_values_by_class(test_df)

            # Impute outliers using IQR method
            train_df = self._impute_outliers_with_iqr(train_df)
            test_df = self._impute_outliers_with_iqr(test_df)

            # Feature engineering
            train_df = self._feature_engineering(train_df)
            test_df = self._feature_engineering(test_df)

            # Split features and target
            input_feature_train_df = train_df.drop(columns=[self.target_column], axis=1)
            target_feature_train_df = train_df[self.target_column]

            input_feature_test_df = test_df.drop(columns=[self.target_column], axis=1)
            target_feature_test_df = test_df[self.target_column]

            # Apply transformations
            preprocessor = self.get_data_transformer_object()
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

            # Ensure directories exist before saving files
            self._ensure_dir(self.data_transformation_config.transformed_train_file_path)
            self._ensure_dir(self.data_transformation_config.transformed_test_file_path)
            self._ensure_dir(self.data_transformation_config.transformed_object_file_path)

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info(f"Data transformation completed for {self.disease_name} successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                disease_name=self.disease_name 
            )

        except Exception as e:
            raise MyException(e, sys) from e