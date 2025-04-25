import sys

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from src.entity.config_entity import ModelPusherConfig
from src.entity.s3_estimator import DiseaseModelEstimator


class ModelPusher:
    def __init__(self, disease_name: str,
                 model_evaluation_artifact: ModelEvaluationArtifact,
                 model_pusher_config: ModelPusherConfig):
        """
        :param disease_name: Name of the disease model being pushed (e.g., 'diabetes', 'heart', 'kidney')
        :param model_evaluation_artifact: Output reference of data evaluation artifact stage
        :param model_pusher_config: Configuration for model pusher
        """
        self.disease_name = disease_name
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.proj1_estimator = DiseaseModelEstimator(
            bucket_name=self.model_pusher_config.bucket_name,
            model_path=self.model_pusher_config.s3_model_key_path,
            disease_name=self.disease_name  # Pass disease name to the S3 Estimator
        )

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Uploads the trained model to S3 for the specific disease.
        """
        logging.info(f"[{self.disease_name.upper()}] Starting model push process...")

        try:
            logging.info("Uploading new model to S3 bucket...")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,disease_name=self.disease_name
            )

            logging.info(f"[{self.disease_name.upper()}] Model pushed to S3: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys) from e