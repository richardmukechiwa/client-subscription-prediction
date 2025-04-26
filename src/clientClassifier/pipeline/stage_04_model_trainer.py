from clientClassifier import logger
from clientClassifier.components.model_trainer import ModelTrainer
from clientClassifier.config.configuration import ConfigurationManager

STAGE_NAME = "Model Trainer Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.train_XGBClassifier()  


if __name__ == "__main__":
    try:
        logger.info(f"{'='*20} {STAGE_NAME} {'='*20}")
        model_trainer_pipeline = ModelTrainerTrainingPipeline()
        model_trainer_pipeline.main()
        logger.info(f"{'='*20} {STAGE_NAME} completed {'='*20}")
    except Exception as e:
        logger.exception(e)
        raise e
