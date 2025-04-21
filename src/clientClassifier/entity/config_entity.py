from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    """Data Ingestion Configuration"""

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Data Validation Configuration"""

    root_dir: Path
    unzip_dir: Path
    STATUS_FILE: str
    all_schema: dict


@dataclass(frozen=True)
class DataTransformationConfig:
    """Data Transformation Configuration"""

    root_dir: Path
    data_path: Path


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    preprocessor_name: str
    label_encoder_names: str
    sm_label_encoder: str
    sm_model_name: str
    sm_processor_name: str
    target_column: str
    class_weight: str
    C: float
    random_state: int
    max_iter: int
    penalty: str
    solver: str
    sm_model_pipeline_name: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    preprocessor_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    mlflow_uri: str
    label_en: Path
    sm_model: Path
