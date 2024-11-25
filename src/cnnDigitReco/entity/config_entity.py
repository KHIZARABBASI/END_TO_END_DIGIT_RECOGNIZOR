from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    unzip_dir: Path
    source_dir: str


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    input_data_file: Path
    output_data_file: Path
    training_dir: Path
    valid_dir: Path


@dataclass(frozen = True)
class TrainingConfig:
    root_dir : Path
    trained_model_path : Path