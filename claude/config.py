"""
Centralized configuration management for the wildlife camera system.

This module provides a unified configuration interface for all components:
- FastAPI camera server
- Motion storage system
- Optical flow analyzer
- Storage server

Configuration is loaded from:
1. Default values (defined in this module)
2. YAML configuration file (config.yaml)
3. Environment variables (override file settings)
4. Command-line arguments (override everything)
"""

import os
import yaml
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Camera hardware and stream settings"""
    # LIVE FEED RESOLUTION (shown in web UI)
    # For 1080p: width=1920, height=1080
    # For 720p: width=1280, height=720
    # For performance: width=640, height=480
    width: int = 640
    height: int = 480
    frame_rate: int = 30
    rotation: int = 0

    # Client connection settings
    max_clients: int = 10
    client_timeout: int = 30  # seconds

    # Display settings
    show_timestamp: bool = True
    timestamp_position: str = 'bottom-right'  # top-left, top-right, bottom-left, bottom-right
    timestamp_color: Tuple[int, int, int] = (255, 255, 255)
    timestamp_size: float = 0.7


@dataclass
class MotionDetectionConfig:
    """Motion detection algorithm settings"""
    enabled: bool = True
    threshold: int = 25  # Lower = more sensitive
    min_area: int = 500  # Minimum pixel area to consider as motion
    blur_kernel_size: int = 21
    highlight_motion: bool = True
    history_size: int = 50  # Number of motion events to keep in memory


@dataclass
class OpticalFlowConfig:
    """Optical flow analysis settings"""
    enabled: bool = True

    # Feature detection parameters
    feature_max: int = 100
    min_distance: int = 7
    quality_level: float = 0.3

    # Grid and binning
    grid_size: Tuple[int, int] = (8, 8)
    direction_bins: int = 8

    # Performance optimization
    frame_skip: int = 2  # Process every Nth frame

    # OPTICAL FLOW PROCESSING RESOLUTION (downscaled for performance)
    # This is NOT the live feed resolution - optical flow analysis uses
    # a downscaled version to reduce CPU load. Keep this low (320x240 or lower).
    max_resolution_width: int = 320
    max_resolution_height: int = 240

    # Visualization (expensive)
    visualization: bool = False
    visualization_scale: float = 1.5

    # Classification settings
    min_features: int = 10
    feature_quality_threshold: float = 0.01
    frame_history: int = 10


@dataclass
class StorageConfig:
    """Local and remote storage settings"""
    # RAM buffer
    ram_buffer_seconds: int = 30
    max_ram_segments: int = 300

    # Local storage
    local_storage_path: str = "./motion_events"
    max_disk_usage_mb: int = 1000
    min_motion_duration_sec: int = 3

    # Remote storage
    remote_storage_url: str = "http://192.168.1.100:8080/storage"
    remote_api_key: str = "your_api_key_here"

    # Transfer settings
    upload_throttle_kbps: int = 500  # Set to 0 to disable uploads
    chunk_upload: bool = True
    transfer_retry_interval_sec: int = 60
    transfer_schedule_active: bool = True
    transfer_schedule_start: int = 1  # 1 AM
    transfer_schedule_end: int = 5    # 5 AM

    # Thumbnail generation (saved with motion events)
    # THUMBNAIL RESOLUTION (for preview images saved to disk)
    # These are small preview images, not the live feed or recorded video.
    generate_thumbnails: bool = True
    thumbnail_width: int = 320
    thumbnail_height: int = 240
    thumbnails_per_event: int = 3

    # WiFi monitoring
    wifi_monitoring: bool = True
    wifi_adapter: str = "wlan0"
    wifi_signal_threshold_low: int = -75
    wifi_signal_threshold_good: int = -65
    wifi_throttle_poor: int = 100
    wifi_throttle_medium: int = 300
    wifi_throttle_good: int = 800

    # Optical flow storage settings (for backward compatibility with motion_storage.py)
    # Note: These duplicate optical_flow_storage settings but are accessed via storage_config
    store_optical_flow_data: bool = True
    optical_flow_signature_dir: str = "flow_signatures"
    optical_flow_database_path: str = "motion_patterns.db"
    motion_classification_enabled: bool = True
    min_classification_confidence: float = 0.5
    save_flow_visualizations: bool = True


@dataclass
class OpticalFlowStorageConfig:
    """Optical flow pattern storage settings"""
    store_data: bool = True
    signature_dir: str = "flow_signatures"
    database_path: str = "motion_patterns.db"
    classification_enabled: bool = True
    min_classification_confidence: float = 0.5
    save_visualizations: bool = True


@dataclass
class ServerConfig:
    """FastAPI server settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    reload: bool = False


@dataclass
class StorageServerConfig:
    """Storage server settings (for receiving uploads)"""
    host: str = "0.0.0.0"
    port: int = 8080
    storage_path: str = "./received_events"
    max_storage_gb: int = 50
    require_api_key: bool = True
    api_keys: list = field(default_factory=lambda: ["your_api_key_here"])
    enable_chunked_uploads: bool = True
    chunk_size_mb: int = 5
    cleanup_incomplete_hours: int = 24


@dataclass
class WildlifeCameraConfig:
    """Complete wildlife camera system configuration"""
    camera: CameraConfig = field(default_factory=CameraConfig)
    motion_detection: MotionDetectionConfig = field(default_factory=MotionDetectionConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    optical_flow_storage: OpticalFlowStorageConfig = field(default_factory=OpticalFlowStorageConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    storage_server: StorageServerConfig = field(default_factory=StorageServerConfig)


class ConfigLoader:
    """Load and manage configuration from multiple sources"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration loader.

        Args:
            config_file: Path to YAML config file. If None, looks for:
                        - ./config.yaml
                        - ~/.wildlife-camera/config.yaml
                        - /etc/wildlife-camera/config.yaml
        """
        self.config_file = self._find_config_file(config_file)
        self.config = WildlifeCameraConfig()

    def _find_config_file(self, config_file: Optional[str]) -> Optional[Path]:
        """Find configuration file in standard locations"""
        if config_file:
            path = Path(config_file)
            if path.exists():
                return path
            logger.warning(f"Specified config file not found: {config_file}")
            return None

        # Search standard locations
        search_paths = [
            Path("./config.yaml"),
            Path.home() / ".wildlife-camera" / "config.yaml",
            Path("/etc/wildlife-camera/config.yaml"),
        ]

        for path in search_paths:
            if path.exists():
                logger.info(f"Found config file: {path}")
                return path

        logger.info("No config file found, using defaults")
        return None

    def load(self) -> WildlifeCameraConfig:
        """
        Load configuration from file and environment variables.

        Priority (highest to lowest):
        1. Environment variables
        2. Config file
        3. Defaults
        """
        # Start with defaults
        config = WildlifeCameraConfig()

        # Load from file if available
        if self.config_file:
            config = self._load_from_file(self.config_file, config)

        # Override with environment variables
        config = self._load_from_env(config)

        # Validate configuration
        self._validate(config)

        self.config = config
        return config

    def _load_from_file(self, config_file: Path, config: WildlifeCameraConfig) -> WildlifeCameraConfig:
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)

            if not data:
                logger.warning(f"Config file is empty: {config_file}")
                return config

            # Update configuration sections
            if 'camera' in data:
                self._update_dataclass(config.camera, data['camera'])
            if 'motion_detection' in data:
                self._update_dataclass(config.motion_detection, data['motion_detection'])
            if 'optical_flow' in data:
                self._update_dataclass(config.optical_flow, data['optical_flow'])
            if 'storage' in data:
                self._update_dataclass(config.storage, data['storage'])
            if 'optical_flow_storage' in data:
                self._update_dataclass(config.optical_flow_storage, data['optical_flow_storage'])
            if 'server' in data:
                self._update_dataclass(config.server, data['server'])
            if 'storage_server' in data:
                self._update_dataclass(config.storage_server, data['storage_server'])

            logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.error(f"Error loading config file {config_file}: {e}")
            logger.info("Using default configuration")

        return config

    def _update_dataclass(self, obj: Any, data: Dict[str, Any]):
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle tuple conversion (YAML loads as list)
                current_value = getattr(obj, key)
                if isinstance(current_value, tuple) and isinstance(value, list):
                    value = tuple(value)
                setattr(obj, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def _load_from_env(self, config: WildlifeCameraConfig) -> WildlifeCameraConfig:
        """Load configuration overrides from environment variables"""
        # Camera settings
        self._env_override(config.camera, 'width', 'WC_CAMERA_WIDTH', int)
        self._env_override(config.camera, 'height', 'WC_CAMERA_HEIGHT', int)
        self._env_override(config.camera, 'frame_rate', 'WC_CAMERA_FRAME_RATE', int)

        # Motion detection
        self._env_override(config.motion_detection, 'enabled', 'WC_MOTION_ENABLED', bool)
        self._env_override(config.motion_detection, 'threshold', 'WC_MOTION_THRESHOLD', int)

        # Optical flow
        self._env_override(config.optical_flow, 'enabled', 'WC_OPTICAL_FLOW_ENABLED', bool)
        self._env_override(config.optical_flow, 'frame_skip', 'WC_OPTICAL_FLOW_FRAME_SKIP', int)

        # Storage
        self._env_override(config.storage, 'local_storage_path', 'WC_STORAGE_PATH', str)
        self._env_override(config.storage, 'remote_storage_url', 'WC_REMOTE_STORAGE_URL', str)
        self._env_override(config.storage, 'remote_api_key', 'WC_REMOTE_API_KEY', str)
        self._env_override(config.storage, 'upload_throttle_kbps', 'WC_UPLOAD_THROTTLE', int)

        # Server
        self._env_override(config.server, 'host', 'WC_SERVER_HOST', str)
        self._env_override(config.server, 'port', 'WC_SERVER_PORT', int)
        self._env_override(config.server, 'log_level', 'WC_LOG_LEVEL', str)

        # Storage server
        self._env_override(config.storage_server, 'host', 'WC_STORAGE_SERVER_HOST', str)
        self._env_override(config.storage_server, 'port', 'WC_STORAGE_SERVER_PORT', int)
        self._env_override(config.storage_server, 'storage_path', 'WC_STORAGE_SERVER_PATH', str)

        return config

    def _env_override(self, obj: Any, attr: str, env_var: str, type_func: type):
        """Override configuration from environment variable"""
        value = os.environ.get(env_var)
        if value is not None:
            try:
                if type_func == bool:
                    # Handle boolean conversion
                    converted = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted = type_func(value)
                setattr(obj, attr, converted)
                logger.debug(f"Override from env: {env_var}={converted}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid value for {env_var}: {value} ({e})")

    def _validate(self, config: WildlifeCameraConfig):
        """Validate configuration values"""
        errors = []

        # Validate camera settings
        if config.camera.width <= 0 or config.camera.height <= 0:
            errors.append("Camera width and height must be positive")
        if config.camera.frame_rate <= 0:
            errors.append("Frame rate must be positive")
        if config.camera.rotation not in (0, 90, 180, 270):
            errors.append("Rotation must be 0, 90, 180, or 270")

        # Validate motion detection
        if config.motion_detection.threshold < 1 or config.motion_detection.threshold > 100:
            errors.append("Motion threshold must be between 1 and 100")

        # Validate storage paths
        storage_path = Path(config.storage.local_storage_path)
        if not storage_path.parent.exists():
            logger.warning(f"Storage path parent directory doesn't exist: {storage_path.parent}")

        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")

    def save(self, output_file: Optional[str] = None):
        """
        Save current configuration to YAML file.

        Args:
            output_file: Path to output file. If None, uses loaded config file path.
        """
        output_path = Path(output_file) if output_file else self.config_file
        if not output_path:
            output_path = Path("./config.yaml")

        # Convert config to dictionary
        config_dict = {
            'camera': asdict(self.config.camera),
            'motion_detection': asdict(self.config.motion_detection),
            'optical_flow': asdict(self.config.optical_flow),
            'storage': asdict(self.config.storage),
            'optical_flow_storage': asdict(self.config.optical_flow_storage),
            'server': asdict(self.config.server),
            'storage_server': asdict(self.config.storage_server),
        }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {output_path}")


# Global configuration instance (lazy loaded)
_global_config: Optional[WildlifeCameraConfig] = None


def get_config(config_file: Optional[str] = None, reload: bool = False) -> WildlifeCameraConfig:
    """
    Get the global configuration instance.

    Args:
        config_file: Path to config file (only used on first call or if reload=True)
        reload: Force reload of configuration

    Returns:
        WildlifeCameraConfig instance
    """
    global _global_config

    if _global_config is None or reload:
        loader = ConfigLoader(config_file)
        _global_config = loader.load()

    return _global_config


def generate_default_config(output_file: str = "config.yaml"):
    """
    Generate a default configuration file with all options documented.

    Args:
        output_file: Path to output configuration file
    """
    config = WildlifeCameraConfig()
    loader = ConfigLoader()
    loader.config = config
    loader.save(output_file)
    print(f"Default configuration generated: {output_file}")
