#!/usr/bin/env python3
"""Quick import and instantiation test for optical flow components."""

print("Testing imports...")
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase
print("✓ Classes import successfully")

print("\nTesting analyzer instantiation...")
analyzer = OpticalFlowAnalyzer()
print("✓ Analyzer instantiated")

print("\nTesting database instantiation...")
db = MotionPatternDatabase(':memory:', '/tmp')
print("✓ Database instantiated")

print("\n✓ All import and instantiation tests passed!")
