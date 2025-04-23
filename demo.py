# below code is to check the logging config
from src.logger import logging

# logging.debug("This is a debug message.")
# logging.info("This is an info message.")
# logging.warning("This is a warning message.")
# logging.error("This is an error message.")
# logging.critical("This is a critical message.")

# --------------------------------------------------------------------------------

# # below code is to check the exception config
# from src.logger import logging
# from src.exception import MyException
# import sys

# try:
#     a = 1+'Z'
# except Exception as e:
#     logging.info(e)
#     raise MyException(e, sys) from e

# --------------------------------------------------------------------------------
from src.logger import logging

from src.diseases.diabetes.pipeline.diabetes_training_pipeline import DiabetesTrainPipeline
from src.diseases.heart.pipeline.heart_training_pipeline import HeartTrainPipeline
from src.diseases.kidney.pipeline.kidney_training_pipeline import KidneyTrainPipeline
from src.entity.config_entity import training_pipeline_config  # Import the existing configuration

def run_all_pipelines():
    logging.info("Starting Diabetes Pipeline...")
    diabetes_pipeline = DiabetesTrainPipeline(training_pipeline_config=training_pipeline_config)
    diabetes_pipeline.run_pipeline()
    logging.info("Diabetes Pipeline completed.")

    logging.info("Starting Heart Disease Pipeline...")
    heart_pipeline = HeartTrainPipeline(training_pipeline_config=training_pipeline_config)
    heart_pipeline.run_pipeline()
    logging.info("Heart Disease Pipeline completed.")

    logging.info("Starting Kidney Disease Pipeline...")
    kidney_pipeline = KidneyTrainPipeline(training_pipeline_config=training_pipeline_config)
    kidney_pipeline.run_pipeline()
    logging.info("Kidney Disease Pipeline completed.")

    logging.info("All pipelines executed successfully.")

if __name__ == "__main__":
    run_all_pipelines()