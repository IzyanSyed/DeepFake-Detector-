from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DFD_")

    spatial_artifacts_checkpoint_path: str | None = None
    fusion_weight_pixel: float = 0.6
    fusion_weight_frequency: float = 0.4
    face_detection_min_confidence: float = 0.5
    frame_sample_rate: int = 5