import os
from datetime import date

# For MongoDB connection
DATABASE_NAME = "Proj1"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = "multi_disease_pipeline"
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"
CURRENT_YEAR = date.today().year

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

# Example disease configurations with algorithm and hyperparameters
DISEASES = {
    "diabetes": {
        "file_name": "diabetes.csv",
        "train_file_name": "train_diabetes.csv",
        "test_file_name": "test_diabetes.csv",
        "collection_name": "diabetes_data",
        "model_file_name": "diabetes_model.pkl",
        "model_bucket_name": "diabetes-model-bucket",
        "target_column": "Outcome",
        "preprocessing_object_file_name": f"diabetes_{PREPROCESSING_OBJECT_FILE_NAME}",
        "algorithm": "gradient_boosting",
        "hyperparameters": {
            "n_estimators": 180,
            "learning_rate": 0.1,
            "loss": "exponential"
        }
    },
    "heart": {
        "file_name": "heart_disease.csv",
        "train_file_name": "train_heart_disease.csv",
        "test_file_name": "test_heart_disease.csv",
        "collection_name": "heart_data",
        "model_file_name": "heart_model.pkl",
        "model_bucket_name": "heart-disease-model-bucket",
        "target_column": "target",
        "preprocessing_object_file_name": f"heart_{PREPROCESSING_OBJECT_FILE_NAME}",
        "algorithm": "random_forest",
        "hyperparameters": {
            "criterion": 'gini',
            "n_estimators": 180,
            "max_depth": 7,
            "max_features": "sqrt",
            "min_samples_leaf": 2,
            "min_samples_split" : 4
        }
    },
    "kidney": {
        "file_name": "kidney_disease.csv",
        "train_file_name": "train_kidney_disease.csv",
        "test_file_name": "test_kidney_disease.csv",
        "collection_name": "kidney_data",
        "model_file_name": "kidney_model.pkl",
        "model_bucket_name": "kidney-disease-model-bucket",
        "target_column": "classification",
        "preprocessing_object_file_name": f"kidney_{PREPROCESSING_OBJECT_FILE_NAME}",
        "algorithm": "gradient_boosting",
        "hyperparameters": {
            "n_estimators": 180,
            "learning_rate": 0.1,
            "loss": "exponential"
        }
    }
}

# General constants for data ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

# Data Validation related constants
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

# Data Transformation related constants
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

# Model Trainer related constants
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS = 180
LOSS: str = 'exponential'
LEARNING_RATE: float = 0.1

# Model Evaluation related constants
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-mlopsproj1"
MODEL_PUSHER_S3_KEY = "model-registry"

# Application settings
APP_HOST = "0.0.0.0"
APP_PORT = 5000
