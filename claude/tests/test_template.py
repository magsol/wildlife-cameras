#!/usr/bin/python3

"""
Simple script to test the HTML template formatting.
This will import the HTML_TEMPLATE from the fastapi_mjpeg_server_with_storage module
and try to format it with sample data to see if it works.
"""

import sys
import os

# Set environment variable to skip picamera2
os.environ["SKIP_PICAMERA"] = "1"

# Import test helpers first to set up mocks
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tests.test_helpers import setup_picamera2_mocks
setup_picamera2_mocks()

# Now import the HTML_TEMPLATE
from fastapi_mjpeg_server_with_storage import HTML_TEMPLATE

# Sample data for formatting
sample_data = {
    "width": 640,
    "height": 480,
    "frame_rate": 30,
    "connection_count": 0,
    "timestamp_checked": "checked",
    "motion_checked": "checked",
    "highlight_checked": "checked",
    "motion_threshold": 25,
    "ts_pos_tl": "",
    "ts_pos_tr": "",
    "ts_pos_bl": "",
    "ts_pos_br": "selected"
}

try:
    # Try to format the template with the sample data
    formatted_html = HTML_TEMPLATE.format(**sample_data)
    print("SUCCESS: Template formatted correctly!")
    
    # Write the formatted HTML to a file for inspection
    with open("test_template_output.html", "w") as f:
        f.write(formatted_html)
    print("Template written to test_template_output.html")
    
except Exception as e:
    print(f"ERROR: Template formatting failed with error: {e}")
    sys.exit(1)