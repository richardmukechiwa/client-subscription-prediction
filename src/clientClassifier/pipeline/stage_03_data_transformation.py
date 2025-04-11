from clientClassifier.config.configuration import ConfigurationManager
from clientClassifier.components.data_transformation import DataTransformation
from clientClassifier import logger     
from pathlib import Path

STAGE_NAME = "Data Transformation stage"

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
         
    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
                
            if status == "False":
                raise Exception("Data Validation failed. Cannot proceed with Data Transformation.") 
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            print(e)
            raise
        
        else:
            logger.info("Data Validation passed. Proceeding with Data Transformation.") 
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()    
            data_transformation = DataTransformation(config=data_transformation_config)
            data_transformation.initiate_data_transformation()
            df = data_transformation.load_data()  # Assuming a method to load data exists
            data_transformation.split_data(df)
            
            
if __name__ == '__main__':
    try:
        logger.info(f"{'>>'*20}stage {STAGE_NAME} started {'<<'*20}")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f"{'>>'*20} stage {STAGE_NAME} completed {'<<'*20}")
    except Exception as e:
        logger.exception(e) 
        raise e
        # raise e   
            
            
    
    
    
    