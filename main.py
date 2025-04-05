from clientClassifier import logger
from clientClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline


STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
except Exception as e:
    logger.exception(e)
    raise e
