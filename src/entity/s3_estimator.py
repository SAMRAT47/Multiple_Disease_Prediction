import sys
from pandas import DataFrame
from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.entity.estimator import MyModel


class DiseaseModelEstimator:
    """
    Handles saving/loading a disease-specific model to/from S3 and performing predictions.
    """

    def __init__(self, bucket_name: str, model_path: str, disease_name: str):
        """
        :param bucket_name: S3 bucket name
        :param model_path: Path to the model in S3
        :param disease_type: Name of the disease (for logging/debugging)
        """
        self.bucket_name = bucket_name
        self.model_path = model_path
        self.disease_name = disease_name
        self.s3 = SimpleStorageService()
        self.loaded_model: MyModel = None

    def is_model_present(self, model_path: str) -> bool:
        try:
            return self.s3.s3_key_path_available(bucket_name=self.bucket_name, s3_key=model_path)
        except MyException as e:
            print(e)
            return False

    def load_model(self) -> MyModel:
        """
        Load the model from S3 and wrap it in MyModel if needed.
        """
        try:
            loaded_model = self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

            if isinstance(loaded_model, dict):
                preprocessing_object = loaded_model.get('preprocessing_object')
                trained_model_object = loaded_model.get('trained_model_object')

                if preprocessing_object is None or trained_model_object is None:
                    raise MyException("Loaded model missing necessary components", sys)

                return MyModel(preprocessing_object=preprocessing_object, trained_model_object=trained_model_object)

            elif isinstance(loaded_model, MyModel):
                return loaded_model

            else:
                raise MyException("Model loaded is not in expected format", sys)

        except Exception as e:
            raise MyException(e, sys)

    def save_model(self, from_file: str, remove: bool = False) -> None:
        """
        Save the local model file to the S3 bucket.
        """
        try:
            self.s3.upload_file(from_file, to_filename=self.model_path, bucket_name=self.bucket_name, remove=remove)
        except Exception as e:
            raise MyException(e, sys)

    @property
    def model(self) -> MyModel:
        if self.loaded_model is None:
            self.loaded_model = self.load_model()
        return self.loaded_model

    def predict(self, dataframe: DataFrame):
        """
        Predict using the loaded model.
        """
        try:
            return self.model.predict(dataframe=dataframe)
        except Exception as e:
            raise MyException(e, sys)