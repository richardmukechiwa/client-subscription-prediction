from clientClassifier.config.configuration import ConfigurationManager
from clientClassifier.components.data_validation import DataValidation      
from clientClassifier import logger 

STAGE_NAME = "Data Validation stage"

class  DataValidationTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        
        data_validation = DataValidation(config=data_validation_config)
        
        #validating all columns
        data_validation.validate_all_columns()
        
        #validating data types
        data_validation.validate_data_types()
            
        
if __name__ == '__main__':
    try:
        logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
    except Exception as e:
        logger.exception(e)      