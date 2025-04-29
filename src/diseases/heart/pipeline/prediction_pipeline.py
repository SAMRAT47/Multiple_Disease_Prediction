import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Union, List, Any
import boto3
import joblib
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file, get_schema_file_path, get_target_column
from src.constants import *


@dataclass
class PredictionPipelineConfig:
    """Configuration for the prediction pipeline."""
    # Update to match your actual S3 bucket name
    bucket_name: str = os.getenv("BUCKET_NAME", "my-mlopsproj1")
    
    # Update to match the actual keys used during model push
    diabetes_model_key: str = os.getenv("DIABETES_MODEL_KEY", "diabetes_model.pkl")
    heart_model_key: str = os.getenv("HEART_MODEL_KEY", "heart_model.pkl")
    kidney_model_key: str = os.getenv("KIDNEY_MODEL_KEY", "kidney_model.pkl")


class S3ModelHandler:
    """Handles downloading models from S3 bucket with robust error handling."""
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        logging.info(f"Initializing S3ModelHandler with bucket: {bucket_name}")
        try:
            self.s3_client = boto3.client('s3')
            # Test connection by checking if bucket exists
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logging.info(f"Successfully connected to S3 bucket: {self.bucket_name}")
        except Exception as e:
            logging.error(f"S3 connection error: {str(e)}")
            if "credentials" in str(e).lower():
                logging.error("AWS credentials issue detected. Please check your AWS configuration.")
            elif "NoSuchBucket" in str(e):
                logging.error(f"Bucket '{self.bucket_name}' does not exist.")
            elif "AccessDenied" in str(e):
                logging.error(f"Access denied to bucket '{self.bucket_name}'.")
            else:
                logging.error(f"Other S3 error: {str(e)}")
            self.s3_client = None
        
    def download_model(self, model_key: str, local_path: str) -> bool:
        """
        Downloads model from S3 to local path with robust error handling.
        Returns True if successful, False otherwise.
        """
        logging.info(f"Attempting to download model: s3://{self.bucket_name}/{model_key} to {local_path}")
        
        if self.s3_client is None:
            logging.error("S3 client not initialized. Cannot download model.")
            return False
        
        try:
            # Create directory for the local path if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # First check if the model exists in S3
            try:
                response = self.s3_client.head_object(Bucket=self.bucket_name, Key=model_key)
                model_size = response['ContentLength']
                logging.info(f"Model found in S3: s3://{self.bucket_name}/{model_key} (Size: {model_size} bytes)")
            except Exception as e:
                logging.error(f"Model not found in S3: s3://{self.bucket_name}/{model_key}")
                logging.error(f"Error details: {str(e)}")
                return False
            
            # Download the model
            logging.info(f"Downloading model from S3...")
            self.s3_client.download_file(self.bucket_name, model_key, local_path)
            
            # Verify the file was downloaded
            if os.path.exists(local_path):
                local_size = os.path.getsize(local_path)
                logging.info(f"Model downloaded successfully to {local_path} (Size: {local_size} bytes)")
                return True
            else:
                logging.error(f"Download completed but file not found at {local_path}")
                return False
                
        except Exception as e:
            logging.error(f"Error downloading model from S3: {str(e)}")
            return False


class BaseDiseasePredictor:
    """Base class for disease prediction with common functionality."""
    def __init__(self, disease_name: str, model_key: str, bucket_name: str):
        self.disease_name = disease_name
        self.model_key = model_key
        self.bucket_name = bucket_name
        self.model_dir = os.path.join(os.getcwd(), "saved_models")
        self.local_model_path = os.path.join(self.model_dir, f"{disease_name}_model.pkl")
        self.schema_path = get_schema_file_path(disease_name)
        self._schema_config = read_yaml_file(self.schema_path)
        self.model = self._load_model()
    
    def _load_model(self):
        """Load model from local path or download from S3 if not available."""
        try:
            # Create model directory if it doesn't exist
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Try to load the model from local path
            if os.path.exists(self.local_model_path):
                logging.info(f"Loading model from local path: {self.local_model_path}")
                return joblib.load(self.local_model_path)
            
            # If model doesn't exist locally, download it from S3
            logging.info(f"Model not found locally. Downloading from S3...")
            s3_handler = S3ModelHandler(self.bucket_name)
            if s3_handler.download_model(self.model_key, self.local_model_path):
                return joblib.load(self.local_model_path)
            else:
                raise Exception(f"Failed to download model from S3 for {self.disease_name}")
        except Exception as e:
            raise MyException(e, sys)
    
    def _replace_zeros(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace zeros with NaN in specified columns."""
        try:
            cols = self._schema_config.get("columns_to_replace_zeros", [])
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].replace(0, np.nan)
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using SimpleImputer with mean strategy."""
        try:
            imputer = SimpleImputer(strategy='mean')
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if not numeric_columns.empty:
                df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _impute_outliers_with_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute outliers using IQR method."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with Q1 or Q3
                df[col] = np.where(df[col] < lower_bound, Q1, df[col])
                df[col] = np.where(df[col] > upper_bound, Q3, df[col])
            
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _preprocess_data method")
    
    def predict(self, data: Dict[str, Any]) -> int:
        """Make prediction based on input data."""
        try:
            preprocessed_data = self._preprocess_data(data)
            prediction = self.model.predict(preprocessed_data)
            return int(prediction[0])
        except Exception as e:
            raise MyException(e, sys)


class DiabetesPredictor(BaseDiseasePredictor):
    """Diabetes disease prediction pipeline."""
    def __init__(self, model_key: str, bucket_name: str):
        super().__init__("diabetes", model_key, bucket_name)
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering specific to diabetes."""
        try:
            # Extract BMI for derived features
            bmi = df['BMI'].iloc[0] if 'BMI' in df.columns else 0
            
            # Create BMI categories
            df['NewBMI_Underweight'] = 1 if bmi <= 18.5 else 0
            df['NewBMI_Normal'] = 1 if 18.5 < bmi <= 24.9 else 0
            df['NewBMI_Overweight'] = 1 if 24.9 < bmi <= 29.9 else 0
            df['NewBMI_Obesity_type1'] = 1 if 29.9 < bmi <= 34.9 else 0
            df['NewBMI_Obesity_type2'] = 1 if bmi > 34.9 else 0
            
            # Create Insulin categories
            insulin = df['Insulin'].iloc[0] if 'Insulin' in df.columns else 0
            df['NewInsulinScore_Normal'] = 1 if 70 <= insulin <= 130 else 0
            
            # Create Glucose categories
            glucose = df['Glucose'].iloc[0] if 'Glucose' in df.columns else 0
            df['NewGlucose_Low'] = 1 if glucose <= 70 else 0
            df['NewGlucose_Normal'] = 1 if 70 < glucose <= 99 else 0
            df['NewGlucose_Overweight'] = 1 if 99 < glucose <= 125 else 0
            df['NewGlucose_Secret'] = 1 if 125 < glucose <= 200 else 0
            df['NewGlucose_High'] = 1 if glucose > 200 else 0
            
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess diabetes data."""
        try:
            # Convert input dictionary to dataframe
            df = pd.DataFrame([data])
            
            # Apply preprocessing steps
            df = self._replace_zeros(df)
            df = self._impute_missing_values(df)
            df = self._impute_outliers_with_iqr(df)
            df = self._feature_engineering(df)
            
            logging.info(f"Preprocessed data shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)


class HeartDiseasePredictor(BaseDiseasePredictor):
    """Heart disease prediction pipeline."""
    def __init__(self, model_key: str, bucket_name: str):
        super().__init__("heart", model_key, bucket_name)
    
    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess heart disease data."""
        try:
            # Convert input dictionary to dataframe
            df = pd.DataFrame([data])
            
            # Apply preprocessing steps
            df = self._impute_missing_values(df)
            df = self._impute_outliers_with_iqr(df)
            
            logging.info(f"Preprocessed heart disease data shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)


class KidneyDiseasePredictor(BaseDiseasePredictor):
    """Kidney disease prediction pipeline."""
    def __init__(self, model_key: str, bucket_name: str):
        super().__init__("kidney", model_key, bucket_name)
    
    def _apply_column_value_replacements(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column value replacements specific to kidney disease."""
        try:
            # Manual replacements for known dirty values
            manual_replacements = {
                'dm': {"\tno": "no", "\tyes": "yes", " yes": "yes", "\t?": np.nan, "??": np.nan},
                'cad': {"\tno": "no", "\t?": np.nan, "??": np.nan},
                'classification': {"ckd\t": "ckd", "notckd": "not ckd", "\t?": np.nan, "??": np.nan}
            }
            
            for col, replacements in manual_replacements.items():
                if col in df.columns:
                    df[col] = df[col].replace(replacements)
            
            # General cleanup: replace '\t??' or '??' in all string columns with NaN
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].replace({'\t?': np.nan, '??': np.nan})
            
            # Apply schema-based mapping values
            column_value_mappings = self._schema_config.get("column_value_mappings", {})
            for col, mapping in column_value_mappings.items():
                if col in df.columns:
                    df[col] = df[col].map(mapping)
            
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _rename_columns_as_per_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns as per schema for kidney disease data."""
        try:
            column_renaming_map = self._schema_config.get("column_renaming_map", {})
            if column_renaming_map:
                df = df.rename(columns=column_renaming_map)
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _label_encode_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns."""
        try:
            categorical_cols = self._schema_config.get("categorical_columns", [])
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_categorical_cols = [renaming_map.get(col, col) for col in categorical_cols]
            existing_categorical_cols = [col for col in renamed_categorical_cols if col in df.columns]
            
            for col in existing_categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
            
            return df
        except Exception as e:
            raise MyException(e, sys)
    
    def _random_sampling_impute(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Impute missing values using random sampling."""
        try:
            missing_count = df[feature].isna().sum()
            if missing_count == 0:
                return df
            
            non_null_values = df[feature].dropna()
            if len(non_null_values) == 0:
                # If no non-null values, use 0 as default
                df[feature] = df[feature].fillna(0)
                return df
                
            if missing_count > len(non_null_values):
                random_sample = non_null_values.sample(missing_count, replace=True, random_state=42)
            else:
                random_sample = non_null_values.sample(missing_count, random_state=42)
            
            random_sample.index = df[df[feature].isna()].index
            df.loc[df[feature].isna(), feature] = random_sample
            return df
        except Exception as e:
            raise MyException(f"Error in random sampling for {feature}: {e}", sys)
    
    def _mode_impute(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Impute missing values using mode."""
        try:
            if feature in df.columns and df[feature].isnull().sum() > 0:
                if len(df[feature].dropna()) > 0:
                    mode = df[feature].mode()[0]
                    df[feature] = df[feature].fillna(mode)
                else:
                    # If all values are null, use a default value
                    if df[feature].dtype == np.number:
                        df[feature] = df[feature].fillna(0)
                    else:
                        df[feature] = df[feature].fillna("unknown")
            return df
        except Exception as e:
            raise MyException(f"Error in mode imputation for {feature}: {e}", sys)
    
    def _preprocess_data(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess kidney disease data."""
        try:
            # Convert input dictionary to dataframe
            df = pd.DataFrame([data])
            
            # Apply preprocessing steps
            df = self._apply_column_value_replacements(df)
            df = self._rename_columns_as_per_schema(df)
            
            # Get numerical columns that exist in the dataframe
            numerical_cols = self._schema_config.get("numerical_columns", [])
            renaming_map = self._schema_config.get("column_renaming_map", {})
            renamed_numerical_cols = [renaming_map.get(col, col) for col in numerical_cols]
            existing_numerical_cols = [col for col in renamed_numerical_cols if col in df.columns]
            
            # Apply imputation for numerical columns
            for col in existing_numerical_cols:
                df = self._random_sampling_impute(df, col)
            
            # Apply imputation for specific domain columns if they exist
            for col in ["red_blood_cells", "pus_cell"]:
                if col in df.columns:
                    df = self._random_sampling_impute(df, col)
            
            # Get categorical columns that exist in the dataframe
            categorical_cols = self._schema_config.get("categorical_columns", [])
            renamed_categorical_cols = [renaming_map.get(col, col) for col in categorical_cols]
            existing_categorical_cols = [col for col in renamed_categorical_cols if col in df.columns]
            
            # Apply mode imputation for categorical columns
            for col in existing_categorical_cols:
                df = self._mode_impute(df, col)
            
            # Apply label encoding for categorical columns
            df = self._label_encode_columns(df)
            
            # Handle outliers in numerical features
            df = self._impute_outliers_with_iqr(df)
            
            logging.info(f"Preprocessed kidney disease data shape: {df.shape}")
            return df
        except Exception as e:
            raise MyException(e, sys)


class PredictionPipeline:
    """Main prediction pipeline class that handles all disease predictions."""
    def __init__(self):
        self.config = PredictionPipelineConfig()
        self.diabetes_predictor = DiabetesPredictor(
            model_key=self.config.diabetes_model_key,
            bucket_name=self.config.bucket_name
        )
        self.heart_predictor = HeartDiseasePredictor(
            model_key=self.config.heart_model_key,
            bucket_name=self.config.bucket_name
        )
        self.kidney_predictor = KidneyDiseasePredictor(
            model_key=self.config.kidney_model_key,
            bucket_name=self.config.bucket_name
        )
    
    def predict_diabetes(self, data: Dict[str, Any]) -> Dict[str, Union[int, str]]:
        """Predict diabetes based on input data."""
        try:
            prediction = self.diabetes_predictor.predict(data)
            return {
                "prediction": prediction,
                "message": "The person is diabetic" if prediction == 1 else "The person is not diabetic"
            }
        except Exception as e:
            logging.error(f"Error in diabetes prediction: {e}")
            raise e
    
    def predict_heart_disease(self, data: Dict[str, Any]) -> Dict[str, Union[int, str]]:
        """Predict heart disease based on input data."""
        try:
            prediction = self.heart_predictor.predict(data)
            return {
                "prediction": prediction,
                "message": "The person has heart disease" if prediction == 1 else "The person does not have heart disease"
            }
        except Exception as e:
            logging.error(f"Error in heart disease prediction: {e}")
            raise e
    
    def predict_kidney_disease(self, data: Dict[str, Any]) -> Dict[str, Union[int, str]]:
        """Predict kidney disease based on input data."""
        try:
            prediction = self.kidney_predictor.predict(data)
            return {
                "prediction": prediction,
                "message": "The person has kidney disease" if prediction == 1 else "The person does not have kidney disease"
            }
        except Exception as e:
            logging.error(f"Error in kidney disease prediction: {e}")
            raise e