from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    disease_name: str
    feature_store_file_path: str
    trained_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    disease_name: str
    validation_status: bool
    message: str
    validation_report_file_path: str


@dataclass
class DataTransformationArtifact:
    disease_name: str
    preprocessed_object_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str


@dataclass
class ClassificationMetricArtifact:
    f1_score: float
    precision_score: float
    recall_score: float


@dataclass
class ModelTrainerArtifact:
    disease_name: str
    trained_model_file_path: str
    metric_artifact: ClassificationMetricArtifact


@dataclass
class ModelEvaluationArtifact:
    disease_name: str
    is_model_accepted: bool
    changed_accuracy: float
    s3_model_path: str
    trained_model_path: str


@dataclass
class ModelPusherArtifact:
    disease_name: str
    bucket_name: str
    s3_model_path: str