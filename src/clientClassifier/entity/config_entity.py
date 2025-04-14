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
    sm_model_name: str
    sm_processor_name: str
    target_column: str
    class_weight: str
    C: float
    random_state: int
    max_iter: int
    penalty: str    
    solver: str