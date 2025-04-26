from clientClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from clientClassifier.utils.common import read_yaml, create_directories
from pathlib import Path
from clientClassifier.entity.config_entity import (DataIngestionConfig,
                                                   DataValidationConfig,
                                                   DataTransformationConfig,
                                                   ModelTrainerConfig,
                                                   ModelEvaluationConfig)    

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL  = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir    
                
            )
        
        return data_ingestion_config
    
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            STATUS_FILE = config.STATUS_FILE,
            unzip_dir = config.unzip_dir,
            all_schema = schema
          
            
                
            )
        
        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
   
        
        # create directories if not exist
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir=  config.root_dir,
            data_path= config.data_path,
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer     
        params = self.params.xgb_classifier
        schema = self.schema.TARGET_COLUMN
        
        

        create_directories([config["root_dir"]])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            model_name=config.model_name,
            preprocessor_name=config.preprocessor_name,
            sm_model_pipeline_name=config.sm_model_pipeline_name,
            label_encoder_names=config.label_encoder_names,
            sm_label_encoder=config.sm_label_encoder,
            sm_model_name=config.sm_model_name,
            sm_processor_name=config.sm_processor_name,
            target_column=schema.name,
            max_depth = params.max_depth,
            #class_weight=params.class_weight,
            n_estimators= params.n_estimators,
            random_state=params.random_state,
            rf_model_name   = config.rf_model_name,
            rf_preprocessor_name=config.rf_preprocessor_name,
            xgb_model_name=config.xgb_model_name,
            xgb_preprocessor_name=config.xgb_preprocessor_name
           
        )

        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config['model_evaluation']
        params = self.params['xgb_classifier']
        schema = self.schema['TARGET_COLUMN']
        
        create_directories([config['root_dir']])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config['root_dir']),
            test_data_path=Path(config['test_data_path']),
            model_path=Path(config['model_path']),
            rf_model = Path(config['rf_model']),
            rf_processor  = Path(config['rf_processor']),
            preprocessor_path = Path(config['preprocessor_path']),
            xgb_encoder= Path(config['xgb_encoder']),
            xgb_model= Path(config['xgb_model']),   
            sm_model = Path(config['sm_model']),
            xgb_processor = Path(config['xgb_processor']),  
            all_params=params,
            metric_file_name=Path(config['metric_file_name']),
            target_column=schema.name,
            mlflow_uri="https://dagshub.com/richardmukechiwa/client-subscription-prediction.mlflow"
        )
        
        return model_evaluation_config