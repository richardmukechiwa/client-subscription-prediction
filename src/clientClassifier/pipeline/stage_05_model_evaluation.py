from clientClassifier import logger
from clientClassifier.components.model_evaluation import ModelEvaluation
from clientClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)  # Correct variable name
        model_evaluation.feature_importance()
        model_evaluation.log_selected_features_into_mlflow()
        
if __name__ == "__main__":
    try:
        logger.info(f"{'-'*30}Model Evaluation Stage Started{'-'*30}")
        model_evaluation_pipeline = ModelEvaluationTrainingPipeline()
        model_evaluation_pipeline.main()
        logger.info(f"{'-'*30}Model Evaluation Stage Completed{'-'*30}")
    except Exception as e:
        logger.exception(e)
        raise e
            
        
        