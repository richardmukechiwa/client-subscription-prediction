from clientClassifier import logger
from clientClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from clientClassifier.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
                                                        


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