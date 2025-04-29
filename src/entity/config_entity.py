import os
from dataclasses import dataclass
from datetime import datetime
from src.constants import *

# Generate the timestamp for artifact directories
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP

    def __post_init__(self):
        self.artifact_dir = os.path.join(ARTIFACT_DIR, self.timestamp)

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    def __init__(self, disease_name: str, training_pipeline_config: TrainingPipelineConfig):
        disease_config = DISEASES.get(disease_name, {})
        if not disease_config:
            raise ValueError(f"Invalid disease name '{disease_name}'")

        self.disease_name = disease_name
        self.file_name = disease_config.get("file_name")
        self.train_file_name = disease_config.get("train_file_name")
        self.test_file_name = disease_config.get("test_file_name")
        self.collection_name = disease_config.get("collection_name")
        self.model_file_name = disease_config.get("model_file_name")
        self.model_bucket_name = disease_config.get("model_bucket_name")
        self.target_column = disease_config.get("target_column")

        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, disease_name, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, self.file_name)
        self.training_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, self.train_file_name)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, self.test_file_name)
        self.train_test_split_ratio = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO

@dataclass
class DataValidationConfig:
    data_validation_dir: str
    validation_report_file_path: str

    def __init__(self, disease_name: str, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, disease_name, DATA_VALIDATION_DIR_NAME)
        self.validation_report_file_path = os.path.join(self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str
    disease_name: str

    def __init__(self, disease_name: str, training_pipeline_config: TrainingPipelineConfig):
        disease_config = DISEASES[disease_name]
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, disease_name, DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, disease_config["train_file_name"].replace(".csv", ".npy"))
        self.transformed_test_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR, disease_config["test_file_name"].replace(".csv", ".npy"))
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, disease_config["preprocessing_object_file_name"])
        self.disease_name = disease_name

@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str
    trained_model_file_path: str
    disease_name: str
    expected_accuracy: float
    model_config_file_path: str
    algorithm: str
    hyperparameters: dict

    def __init__(self, disease_name: str, training_pipeline_config: TrainingPipelineConfig):
        disease_config = DISEASES[disease_name]
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, disease_name, MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path = os.path.join(self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, disease_config["model_file_name"])
        self.expected_accuracy = MODEL_TRAINER_EXPECTED_SCORE
        self.model_config_file_path = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
        self.algorithm = disease_config["algorithm"]
        self.hyperparameters = disease_config.get("hyperparameters", {})
        self.disease_name = disease_name

@dataclass
class ModelEvaluationConfig:
    disease_name: str
    training_pipeline_config: any
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_PUSHER_S3_KEY
    

    def __init__(self, disease_name: str, training_pipeline_config=None):
        disease_config = DISEASES[disease_name]
        self.changed_threshold_score = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
        self.bucket_name = disease_config["model_bucket_name"]
        self.s3_model_key_path = disease_config["model_file_name"]
        self.disease_name = disease_name
        self.training_pipeline_config = training_pipeline_config  # Store the config if needed

@dataclass
class ModelPusherConfig:
    disease_name: str
    training_pipeline_config: TrainingPipelineConfig  # ← Add this
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_PUSHER_S3_KEY

    def __post_init__(self):
        if not self.disease_name:
            raise ValueError("disease_name must be provided.")
        disease_config = DISEASES.get(self.disease_name)
        if disease_config is None:
            raise ValueError(f"Configuration for disease '{self.disease_name}' not found.")
        self.bucket_name = MODEL_BUCKET_NAME
        self.s3_model_key_path = disease_config["model_file_name"]

@dataclass
class DiseasePredictorConfig:
    disease_name: str
    model_file_path: str = ""
    bucket_name: str = MODEL_BUCKET_NAME

    def __post_init__(self):
        disease_config = DISEASES[self.disease_name]
        self.model_file_path = disease_config["model_file_name"]


