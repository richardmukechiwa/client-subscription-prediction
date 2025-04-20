from pathlib import Path

from clientClassifier import logger
from clientClassifier.components.data_ingestion import DataIngestion
from clientClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
    except Exception as e:
        logger.exception(e)
        raise e
