"""
Command-line interface for wildlife camera system.

Provides unified command-line arguments for all configuration options.
"""

import argparse
import sys
from config import get_config, generate_default_config


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all configuration options"""
    parser = argparse.ArgumentParser(
        description="Wildlife Camera Motion Detection System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML)'
    )
    parser.add_argument(
        '--generate-config',
        action='store_true',
        help='Generate default config.yaml and exit'
    )

    # Camera options
    camera_group = parser.add_argument_group('Camera Settings')
    camera_group.add_argument('--width', type=int, help='Camera width')
    camera_group.add_argument('--height', type=int, help='Camera height')
    camera_group.add_argument('--frame-rate', type=int, help='Frame rate (FPS)')
    camera_group.add_argument('--rotation', type=int, choices=[0, 90, 180, 270], help='Camera rotation')

    # Motion detection options
    motion_group = parser.add_argument_group('Motion Detection')
    motion_group.add_argument('--motion-enabled', type=bool, help='Enable motion detection')
    motion_group.add_argument('--motion-threshold', type=int, help='Motion sensitivity (5-100, lower=more sensitive)')
    motion_group.add_argument('--motion-min-area', type=int, help='Minimum motion area (pixels)')

    # Optical flow options
    flow_group = parser.add_argument_group('Optical Flow')
    flow_group.add_argument('--optical-flow-enabled', type=bool, help='Enable optical flow analysis')
    flow_group.add_argument('--optical-flow-frame-skip', type=int, help='Process every Nth frame')
    flow_group.add_argument('--optical-flow-visualization', action='store_true', help='Enable flow visualization')

    # Storage options
    storage_group = parser.add_argument_group('Storage')
    storage_group.add_argument('--storage-path', type=str, help='Local storage directory')
    storage_group.add_argument('--remote-url', type=str, help='Remote storage URL')
    storage_group.add_argument('--remote-api-key', type=str, help='Remote storage API key')
    storage_group.add_argument('--upload-throttle', type=int, help='Upload throttle (KB/s, 0=disabled)')
    storage_group.add_argument('--disable-uploads', action='store_true', help='Disable remote uploads')

    # Server options
    server_group = parser.add_argument_group('Server')
    server_group.add_argument('--host', type=str, help='Server host')
    server_group.add_argument('--port', type=int, help='Server port')
    server_group.add_argument('--log-level', type=str, choices=['debug', 'info', 'warning', 'error'], help='Log level')
    server_group.add_argument('--reload', action='store_true', help='Enable auto-reload for development')

    return parser


def apply_cli_args(config, args):
    """Apply command-line arguments to configuration"""
    # Camera settings
    if args.width is not None:
        config.camera.width = args.width
    if args.height is not None:
        config.camera.height = args.height
    if args.frame_rate is not None:
        config.camera.frame_rate = args.frame_rate
    if args.rotation is not None:
        config.camera.rotation = args.rotation

    # Motion detection
    if args.motion_enabled is not None:
        config.motion_detection.enabled = args.motion_enabled
    if args.motion_threshold is not None:
        config.motion_detection.threshold = args.motion_threshold
    if args.motion_min_area is not None:
        config.motion_detection.min_area = args.motion_min_area

    # Optical flow
    if args.optical_flow_enabled is not None:
        config.optical_flow.enabled = args.optical_flow_enabled
    if args.optical_flow_frame_skip is not None:
        config.optical_flow.frame_skip = args.optical_flow_frame_skip
    if args.optical_flow_visualization:
        config.optical_flow.visualization = True

    # Storage
    if args.storage_path is not None:
        config.storage.local_storage_path = args.storage_path
    if args.remote_url is not None:
        config.storage.remote_storage_url = args.remote_url
    if args.remote_api_key is not None:
        config.storage.remote_api_key = args.remote_api_key
    if args.upload_throttle is not None:
        config.storage.upload_throttle_kbps = args.upload_throttle
    if args.disable_uploads:
        config.storage.upload_throttle_kbps = 0

    # Server
    if args.host is not None:
        config.server.host = args.host
    if args.port is not None:
        config.server.port = args.port
    if args.log_level is not None:
        config.server.log_level = args.log_level
    if args.reload:
        config.server.reload = True

    return config


def load_config_with_cli():
    """
    Load configuration with command-line argument support.

    Priority (highest to lowest):
    1. Command-line arguments
    2. Environment variables
    3. Config file
    4. Defaults

    Returns:
        Tuple of (config, args)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Handle --generate-config
    if args.generate_config:
        generate_default_config()
        sys.exit(0)

    # Load configuration from file and environment
    config = get_config(config_file=args.config)

    # Apply command-line overrides
    config = apply_cli_args(config, args)

    return config, args
