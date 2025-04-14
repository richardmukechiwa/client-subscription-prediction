from clientClassifier import logger
from clientClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from clientClassifier.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from clientClassifier.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from clientClassifier.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline                                                      


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation stage"

try:
    logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")    
except Exception as e:
    logger.exception(e)  
    raise e

STAGE_NAME = "Data Transformation stage"
try:
    logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")    
except Exception as e:
    logger.exception(e)  
    raise e     

STAGE_NAME = "Model Trainer stage"  
try:
    logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
except Exception as e:
    logger.exception(e)  
    raise e 
    




