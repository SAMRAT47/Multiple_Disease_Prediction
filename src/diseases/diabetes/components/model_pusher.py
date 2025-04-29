import sys
import os
import boto3
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
        self.disease_name = disease_name
        self.model_evaluation_artifact = model_evaluation_artifact
        self.model_pusher_config = model_pusher_config

        self.proj1_estimator = DiseaseModelEstimator(
            bucket_name=self.model_pusher_config.bucket_name,
            model_path=self.model_pusher_config.s3_model_key_path,
            disease_name=self.disease_name
        )

    def verify_s3_model(self) -> bool:
        """Verify if the model exists in S3 after pushing"""
        try:
            s3_client = boto3.client('s3')
            s3_client.head_object(
                Bucket=self.model_pusher_config.bucket_name,
                Key=self.model_pusher_config.s3_model_key_path
            )
            return True
        except Exception as e:
            logging.error(f"S3 model verification failed: {str(e)}")
            return False

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Uploads the trained model to S3 for the specific disease with verification.
        """
        logging.info(f"[{self.disease_name.upper()}] Starting model push process...")

        try:
            # Check if the trained model file exists locally
            if not os.path.exists(self.model_evaluation_artifact.trained_model_path):
                raise Exception(f"Trained model not found at {self.model_evaluation_artifact.trained_model_path}")
            
            logging.info(f"Uploading model from {self.model_evaluation_artifact.trained_model_path} to S3 bucket...")
            self.proj1_estimator.save_model(from_file=self.model_evaluation_artifact.trained_model_path)
            
            # Verify the model was pushed correctly
            if not self.verify_s3_model():
                raise Exception(f"Failed to verify the model in S3 after pushing.")

            model_pusher_artifact = ModelPusherArtifact(
                bucket_name=self.model_pusher_config.bucket_name,
                s3_model_path=self.model_pusher_config.s3_model_key_path,
                disease_name=self.disease_name
            )

            logging.info(f"[{self.disease_name.upper()}] Model successfully pushed to S3: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            error_msg = f"Error pushing model to S3: {str(e)}"
            logging.error(error_msg)
            raise MyException(e, sys) from e