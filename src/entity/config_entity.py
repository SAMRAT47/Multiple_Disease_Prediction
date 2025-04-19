import os
from dataclasses import dataclass
from datetime import datetime
from src.constants import *

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = ""  # You can set a dynamic pipeline name if needed.
    artifact_dir: str = os.path.join("artifact", TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


class DataIngestionConfig:
    def __init__(self, disease_name: str):
        disease_config = DISEASES.get(disease_name, {})

        self.disease_name = disease_name
        self.file_name = disease_config.get("file_name", "default.csv")
        self.train_file_name = disease_config.get("train_file_name", "train_default.csv")
        self.test_file_name = disease_config.get("test_file_name", "test_default.csv")
        self.collection_name = disease_config.get("collection_name", "default_collection")
        self.model_file_name = disease_config.get("model_file_name", "default_model.pkl")
        self.model_bucket_name = disease_config.get("model_bucket_name", "default-model-bucket")
        self.target_column = disease_config.get("target_column", "Outcome")

        # Artifact path
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, disease_name, "data_ingestion")

        # Paths needed for ingestion
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir, "feature_store", self.file_name)
        self.training_file_path = os.path.join(self.data_ingestion_dir, "dataset", self.train_file_name)
        self.testing_file_path = os.path.join(self.data_ingestion_dir, "dataset", self.test_file_name)

        # Split ratio
        self.train_test_split_ratio = 0.2


@dataclass
class DataValidationConfig:
    data_validation_dir: str
    validation_report_file_path: str

    def __init__(self, disease_name: str):
        self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_validation")
        self.validation_report_file_path = os.path.join(self.data_validation_dir, "report.yaml")


@dataclass
class DataTransformationConfig:
    data_transformation_dir: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

    def __init__(self, disease_name: str):
        disease_config = DISEASES[disease_name]
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, "data_transformation")
        self.transformed_train_file_path = os.path.join(
            self.data_transformation_dir, "transformed", disease_config["train_file_name"].replace("csv", "npy")
        )
        self.transformed_test_file_path = os.path.join(
            self.data_transformation_dir, "transformed", disease_config["test_file_name"].replace("csv", "npy")
        )
        self.transformed_object_file_path = os.path.join(
            self.data_transformation_dir, "transformed_object", "preprocessing.pkl"
        )


@dataclass
class ModelTrainerConfig:
    model_trainer_dir: str
    trained_model_file_path: str
    expected_accuracy: float
    model_config_file_path: str
    _n_estimators: int
    _learning_rate: float
    _loss: str

    def __init__(self, disease_name: str):
        disease_config = DISEASES[disease_name]
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.trained_model_file_path = os.path.join(
            self.model_trainer_dir, "trained_model", disease_config["model_file_name"]
        )
        self.expected_accuracy = 0.6  # Example
        self.model_config_file_path = "config/model.yaml"
        self._n_estimators = 180
        self._learning_rate = 0.1
        self._loss = "exponential"


@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float
    bucket_name: str
    s3_model_key_path: str

    def __init__(self, disease_name: str):
        disease_config = DISEASES[disease_name]
        self.changed_threshold_score = 0.02  # Example
        self.bucket_name = disease_config["model_bucket_name"]
        self.s3_model_key_path = disease_config["model_file_name"]


@dataclass
class ModelPusherConfig:
    bucket_name: str
    s3_model_key_path: str

    def __init__(self, disease_name: str):
        disease_config = DISEASES[disease_name]
        self.bucket_name = disease_config["model_bucket_name"]
        self.s3_model_key_path = disease_config["model_file_name"]


@dataclass
class DiseasePredictorConfig:
    model_file_path: str
    model_bucket_name: str

    def __init__(self, disease_name: str):
        disease_config = DISEASES[disease_name]
        self.model_file_path = disease_config["model_file_name"]
        self.model_bucket_name = disease_config["model_bucket_name"]