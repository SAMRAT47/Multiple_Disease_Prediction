import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, LabelEncoder
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

    def _feature_engineering(self, kidney_df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Starting feature engineering...")

        if self.disease_name == "diabetes":
            kidney_df['NewBMI'] = pd.cut(
                kidney_df['BMI'],
                bins=[0, 18.5, 25, 30, 35, np.inf],
                labels=['Underweight', 'Normal', 'Overweight', 'Obesity_type1', 'Obesity_type2']
            )
            kidney_df['NewInsulinScore'] = kidney_df['Insulin'].apply(lambda x: "Normal" if 70 <= x <= 130 else "Abnormal")
            kidney_df['NewGlucose'] = pd.cut(
                kidney_df['Glucose'],
                bins=[0, 70, 99, 125, 200, np.inf],
                labels=['Low', 'Normal', 'Overweight', 'Secret', 'High']
            )

        elif self.disease_name == "heart":
            pass
        elif self.disease_name == "kidney":
            pass

        logging.info("Feature engineering done.")
        return kidney_df

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

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Get categorical columns from schema
            categorical_columns = self._schema_config.get('categorical_columns', [])
            
            # Account for column renaming
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_categorical_columns = [renaming_map.get(col, col) for col in categorical_columns]
            
            categorical_pipeline = Pipeline(steps=[
                ("Imputer", SimpleImputer(strategy='most_frequent'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("CategoricalPipeline", categorical_pipeline, renamed_categorical_columns)
                ],
                remainder='passthrough'
            )

            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final preprocessing pipeline constructed successfully.")

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
            logging.exception("Exception in get_data_transformer_object")
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started")

            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Read data
            train_kidney_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_kidney_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Log columns for debugging
            logging.info(f"Actual columns in train_df: {train_kidney_df.columns.tolist()}")

            # Clean data
            train_kidney_df = self._apply_column_value_replacements(train_kidney_df)
            test_kidney_df = self._apply_column_value_replacements(test_kidney_df)
            
            # Rename columns - do this early to ensure correct column names for remaining operations
            train_kidney_df = self._rename_columns_as_per_schema(train_kidney_df)
            test_kidney_df = self._rename_columns_as_per_schema(test_kidney_df)
            
            # Log columns after renaming for debugging
            logging.info(f"Columns after renaming: {train_kidney_df.columns.tolist()}")
            
            # Get numerical columns that exist in the dataframe
            numerical_cols = self._schema_config.get("numerical_columns", [])
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_numerical_cols = [renaming_map.get(col, col) for col in numerical_cols]
            existing_numerical_cols = [col for col in renamed_numerical_cols if col in train_kidney_df.columns]
            
            # Apply imputation for numerical columns
            for col in existing_numerical_cols:
                train_kidney_df = self._random_sampling_impute(train_kidney_df, col)
                test_kidney_df = self._random_sampling_impute(test_kidney_df, col)
            
            # Apply imputation for specific domain columns if they exist
            for col in ["red_blood_cells", "pus_cell"]:  # Using renamed columns here
                if col in train_kidney_df.columns:
                    train_kidney_df = self._random_sampling_impute(train_kidney_df, col)
                    test_kidney_df = self._random_sampling_impute(test_kidney_df, col)
            
            # Get categorical columns that exist in the dataframe
            categorical_cols = self._schema_config.get("categorical_columns", [])
            renamed_categorical_cols = [renaming_map.get(col, col) for col in categorical_cols]
            existing_categorical_cols = [col for col in renamed_categorical_cols if col in train_kidney_df.columns]
            
            # Apply mode imputation for categorical columns
            for col in existing_categorical_cols:
                train_kidney_df = self._mode_impute(train_kidney_df, col)
                test_kidney_df = self._mode_impute(test_kidney_df, col)
            
            # Apply label encoding for categorical columns
            train_kidney_df = self._label_encode_columns(train_kidney_df, existing_categorical_cols)
            test_kidney_df = self._label_encode_columns(test_kidney_df, existing_categorical_cols)
            
            # Verify target column exists
            if self.target_column not in train_kidney_df.columns:
                # Check if the target column needs to be renamed
                renamed_target = renaming_map.get(self.target_column, self.target_column)
                if renamed_target in train_kidney_df.columns:
                    self.target_column = renamed_target
                else:
                    raise ValueError(f"Target column '{self.target_column}' or its renamed version '{renamed_target}' not found in dataframe. Available columns: {train_kidney_df.columns.tolist()}")
            
            # Split features and target
            input_feature_train_df = train_kidney_df.drop(columns=[self.target_column], axis=1)
            target_feature_train_df = train_kidney_df[self.target_column]

            input_feature_test_df = test_kidney_df.drop(columns=[self.target_column], axis=1)
            target_feature_test_df = test_kidney_df[self.target_column]

            # Get preprocessing pipeline
            preprocessor,preprocessing_object_path = self.get_data_transformer_object()
            
            # Apply preprocessing
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Handle imbalanced data
            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            # Combine features and target
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]
            logging.info("Feature-target concatenation done for train-test df.")

            # Ensure directories exist
            self._ensure_dir(self.data_transformation_config.transformed_train_file_path)
            self._ensure_dir(self.data_transformation_config.transformed_test_file_path)

            # Save transformed data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info(f"Data transformation completed for {self.disease_name} disease successfully")
            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                preprocessed_object_file_path=preprocessing_object_path,
                disease_name=self.disease_name
            )

        except Exception as e:
            raise MyException(e, sys) from e