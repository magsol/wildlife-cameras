#!/usr/bin/env python3
"""
Test harness for evaluating optical flow-based motion classification.
This script allows testing and benchmarking of the optical flow analyzer
on recorded video samples or live camera feed.
"""

import cv2
import numpy as np
import os
import time
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from optical_flow_analyzer import OpticalFlowAnalyzer, MotionPatternDatabase
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logger
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optical_flow_test')

class OpticalFlowTester:
    """Test harness for optical flow-based motion analysis."""
    
    def __init__(self, config=None):
        """Initialize the tester with configuration."""
        self.config = {
            'input_dir': 'test_videos',
            'output_dir': 'test_results',
            'db_path': 'test_motion_patterns.db',
            'signature_dir': 'test_signatures',
            'fps_limit': 30,
            'display_scale': 1.0,
            'show_visualization': True,
            'save_results': True,
            'benchmark_mode': False,
            'labeled_data': {},  # Ground truth labels for test videos
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(self.config['signature_dir'], exist_ok=True)
        
        # Initialize analyzer and database
        self.analyzer = OpticalFlowAnalyzer()
        self.db = MotionPatternDatabase(
            db_path=self.config['db_path'],
            signature_dir=self.config['signature_dir']
        )
        
        # Performance metrics
        self.metrics = {
            'processing_times': [],
            'flow_calculation_times': [],
            'classification_times': [],
            'classifications': [],
            'ground_truth': []
        }
        
        # Figure for live plotting
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.line2 = None
        self.bar_chart = None
        
    def test_video(self, video_path, ground_truth=None):
        """
        Test the optical flow analyzer on a video file.
        
        Args:
            video_path: Path to the video file
            ground_truth: Optional ground truth label for the video
            
        Returns:
            Dictionary of test results
        """
        # Check if file exists
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame interval for fps limiting
        target_fps = min(fps, self.config['fps_limit'])
        frame_interval = max(1, int(fps / target_fps))
        
        # Prepare output video writer
        out = None
        if self.config['save_results']:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(self.config['output_dir'], f"{video_name}_flow.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        
        # Reset the analyzer
        self.analyzer.reset()
        
        # Prepare for processing
        prev_frame = None
        frame_idx = 0
        start_time = time.time()
        flow_features_history = []
        
        # Create live plot if visualization is enabled
        if self.config['show_visualization']:
            self._setup_live_plot()
        
        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames according to interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Process frame
            if prev_frame is not None:
                # Measure processing time
                t_start = time.time()
                
                # Simple motion detection using frame differencing
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                frame_delta = cv2.absdiff(prev_gray, frame_gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours and get motion regions
                motion_regions = []
                motion_detected = False
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_regions.append((x, y, w, h))
                        motion_detected = True
                
                # Calculate optical flow only if motion detected
                flow_features = None
                classification = None
                
                if motion_detected:
                    t_flow_start = time.time()
                    flow_features = self.analyzer.extract_flow(prev_frame, frame, motion_regions)
                    t_flow_end = time.time()
                    
                    # Store flow calculation time
                    if flow_features:
                        flow_features_history.append(flow_features)
                        self.metrics['flow_calculation_times'].append(t_flow_end - t_flow_start)
                        
                        # Classify if we have enough frames
                        if len(flow_features_history) >= 5:
                            t_class_start = time.time()
                            
                            # Generate motion signature from flow history
                            signature = self.analyzer.generate_motion_signature(flow_features_history)
                            
                            # Classify motion
                            if signature:
                                classification = self.analyzer.classify_motion(signature)
                                
                                # Store classification
                                if classification:
                                    self.metrics['classifications'].append(classification)
                                    if ground_truth:
                                        self.metrics['ground_truth'].append(ground_truth)
                            
                            t_class_end = time.time()
                            self.metrics['classification_times'].append(t_class_end - t_class_start)
                
                # Visualize results
                if self.config['show_visualization'] and flow_features:
                    vis_frame = self.analyzer.visualize_flow(frame, flow_features)
                    
                    # Add classification label if available
                    if classification:
                        label = classification['label']
                        confidence = classification['confidence']
                        cv2.putText(vis_frame, f"{label} ({confidence:.2f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    vis_frame = frame
                
                # Record total processing time
                t_end = time.time()
                self.metrics['processing_times'].append(t_end - t_start)
                
                # Update live plot
                if self.config['show_visualization']:
                    self._update_live_plot()
                
                # Save output frame
                if out:
                    out.write(vis_frame)
                
                # Display frame
                if self.config['show_visualization']:
                    # Resize for display if needed
                    if self.config['display_scale'] != 1.0:
                        display_width = int(width * self.config['display_scale'])
                        display_height = int(height * self.config['display_scale'])
                        display_frame = cv2.resize(vis_frame, (display_width, display_height))
                    else:
                        display_frame = vis_frame
                    
                    cv2.imshow('Optical Flow Test', display_frame)
                    
                    # Exit on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Update previous frame
            prev_frame = frame.copy()
            frame_idx += 1
            
            # Print progress
            if frame_idx % 30 == 0:
                progress = frame_idx / frame_count * 100
                elapsed = time.time() - start_time
                remaining = elapsed / frame_idx * (frame_count - frame_idx) if frame_idx > 0 else 0
                logger.info(f"Processing: {progress:.1f}% - Elapsed: {elapsed:.1f}s - Remaining: {remaining:.1f}s")
        
        # Clean up
        cap.release()
        if out:
            out.release()
        
        cv2.destroyAllWindows()
        
        # Calculate final metrics
        processing_fps = 1.0 / np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        flow_time_avg = np.mean(self.metrics['flow_calculation_times']) if self.metrics['flow_calculation_times'] else 0
        class_time_avg = np.mean(self.metrics['classification_times']) if self.metrics['classification_times'] else 0
        
        # Calculate classification accuracy if ground truth is available
        accuracy = 0.0
        if ground_truth and self.metrics['classifications'] and self.metrics['ground_truth']:
            correct = sum(1 for i, gt in enumerate(self.metrics['ground_truth']) 
                         if i < len(self.metrics['classifications']) and 
                         self.metrics['classifications'][i]['label'] == gt)
            accuracy = correct / len(self.metrics['ground_truth']) if self.metrics['ground_truth'] else 0.0
        
        # Save detailed results
        if self.config['save_results']:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            results_path = os.path.join(self.config['output_dir'], f"{video_name}_results.json")
            
            # Prepare results for JSON serialization
            classifications_json = []
            for cls in self.metrics['classifications']:
                classifications_json.append({
                    'label': cls['label'],
                    'confidence': cls['confidence'],
                    'alternatives': [{
                        'label': alt['label'],
                        'confidence': alt['confidence']
                    } for alt in cls.get('alternatives', [])]
                })
            
            results = {
                'video': video_path,
                'ground_truth': ground_truth,
                'frames_processed': frame_idx,
                'processing_fps': processing_fps,
                'flow_calculation_time_avg': flow_time_avg,
                'classification_time_avg': class_time_avg,
                'accuracy': accuracy,
                'classifications': classifications_json
            }
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Return summary
        summary = {
            'video': video_path,
            'frames_processed': frame_idx,
            'processing_fps': processing_fps,
            'flow_calculation_time_avg': flow_time_avg,
            'classification_time_avg': class_time_avg,
            'accuracy': accuracy
        }
        
        return summary
    
    def _setup_live_plot(self):
        """Set up live plot for visualizing performance metrics."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Processing time plot
        self.ax1.set_title('Processing Time (ms)')
        self.ax1.set_xlabel('Frame')
        self.ax1.set_ylabel('Time (ms)')
        self.ax1.grid(True)
        self.line1, = self.ax1.plot([], [], 'b-', label='Total Processing')
        self.line2, = self.ax1.plot([], [], 'r-', label='Flow Calculation')
        self.ax1.legend()
        
        # Classification distribution
        self.ax2.set_title('Classification Distribution')
        self.ax2.set_xlabel('Class')
        self.ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.ion()  # Interactive mode
        plt.show()
    
    def _update_live_plot(self):
        """Update the live plot with current metrics."""
        if not self.fig or not self.ax1 or not self.ax2:
            return
        
        # Update processing time plot
        proc_times = np.array(self.metrics['processing_times'][-100:]) * 1000  # Convert to ms
        flow_times = np.array(self.metrics['flow_calculation_times'][-100:]) * 1000  # Convert to ms
        
        x = np.arange(len(proc_times))
        self.line1.set_data(x, proc_times)
        
        if len(flow_times) > 0:
            x_flow = np.arange(len(flow_times))
            self.line2.set_data(x_flow, flow_times)
        
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Update classification distribution
        if self.metrics['classifications']:
            # Count classifications by label
            labels = {}
            for cls in self.metrics['classifications']:
                label = cls['label']
                if label not in labels:
                    labels[label] = 0
                labels[label] += 1
            
            # Clear previous bar chart
            self.ax2.clear()
            self.ax2.set_title('Classification Distribution')
            self.ax2.set_xlabel('Class')
            self.ax2.set_ylabel('Count')
            
            # Create new bar chart
            x = np.arange(len(labels))
            self.ax2.bar(x, list(labels.values()))
            self.ax2.set_xticks(x)
            self.ax2.set_xticklabels(list(labels.keys()))
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def test_camera(self, camera_idx=0, duration=60):
        """
        Test the optical flow analyzer on a live camera feed.
        
        Args:
            camera_idx: Camera index
            duration: Test duration in seconds
            
        Returns:
            Dictionary of test results
        """
        # Open camera
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            logger.error(f"Error opening camera {camera_idx}")
            return None
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Prepare output video writer
        out = None
        if self.config['save_results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.config['output_dir'], f"camera_{camera_idx}_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset the analyzer
        self.analyzer.reset()
        
        # Prepare for processing
        prev_frame = None
        frame_count = 0
        start_time = time.time()
        flow_features_history = []
        
        # Create live plot if visualization is enabled
        if self.config['show_visualization']:
            self._setup_live_plot()
        
        # Process camera frames
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            if prev_frame is not None:
                # Measure processing time
                t_start = time.time()
                
                # Simple motion detection using frame differencing
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                
                frame_delta = cv2.absdiff(prev_gray, frame_gray)
                thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours and get motion regions
                motion_regions = []
                motion_detected = False
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        motion_regions.append((x, y, w, h))
                        motion_detected = True
                
                # Calculate optical flow only if motion detected
                flow_features = None
                classification = None
                
                if motion_detected:
                    t_flow_start = time.time()
                    flow_features = self.analyzer.extract_flow(prev_frame, frame, motion_regions)
                    t_flow_end = time.time()
                    
                    # Store flow calculation time
                    if flow_features:
                        flow_features_history.append(flow_features)
                        self.metrics['flow_calculation_times'].append(t_flow_end - t_flow_start)
                        
                        # Limit history size
                        if len(flow_features_history) > 30:
                            flow_features_history.pop(0)
                        
                        # Classify if we have enough frames
                        if len(flow_features_history) >= 5:
                            t_class_start = time.time()
                            
                            # Generate motion signature from flow history
                            signature = self.analyzer.generate_motion_signature(flow_features_history)
                            
                            # Classify motion
                            if signature:
                                classification = self.analyzer.classify_motion(signature)
                                
                                # Store classification
                                if classification:
                                    self.metrics['classifications'].append(classification)
                            
                            t_class_end = time.time()
                            self.metrics['classification_times'].append(t_class_end - t_class_start)
                
                # Visualize results
                if self.config['show_visualization'] and flow_features:
                    vis_frame = self.analyzer.visualize_flow(frame, flow_features)
                    
                    # Add classification label if available
                    if classification:
                        label = classification['label']
                        confidence = classification['confidence']
                        cv2.putText(vis_frame, f"{label} ({confidence:.2f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    vis_frame = frame
                
                # Record total processing time
                t_end = time.time()
                self.metrics['processing_times'].append(t_end - t_start)
                
                # Update live plot
                if self.config['show_visualization']:
                    self._update_live_plot()
                
                # Save output frame
                if out:
                    out.write(vis_frame)
                
                # Display frame
                if self.config['show_visualization']:
                    # Display elapsed time
                    elapsed = time.time() - start_time
                    remaining = duration - elapsed
                    cv2.putText(vis_frame, f"Elapsed: {elapsed:.1f}s / Remaining: {remaining:.1f}s", 
                               (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('Camera Test', vis_frame)
                    
                    # Exit on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            # Update previous frame
            prev_frame = frame.copy()
            frame_count += 1
            
            # Print progress every second
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                logger.info(f"Camera test: {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
        
        # Clean up
        cap.release()
        if out:
            out.release()
        
        cv2.destroyAllWindows()
        
        # Calculate final metrics
        processing_fps = 1.0 / np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0
        flow_time_avg = np.mean(self.metrics['flow_calculation_times']) if self.metrics['flow_calculation_times'] else 0
        class_time_avg = np.mean(self.metrics['classification_times']) if self.metrics['classification_times'] else 0
        
        # Save detailed results
        if self.config['save_results']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = os.path.join(self.config['output_dir'], f"camera_{camera_idx}_{timestamp}_results.json")
            
            # Prepare results for JSON serialization
            classifications_json = []
            for cls in self.metrics['classifications']:
                classifications_json.append({
                    'label': cls['label'],
                    'confidence': cls['confidence'],
                    'alternatives': [{
                        'label': alt['label'],
                        'confidence': alt['confidence']
                    } for alt in cls.get('alternatives', [])]
                })
            
            results = {
                'camera': camera_idx,
                'frames_processed': frame_count,
                'duration': time.time() - start_time,
                'processing_fps': processing_fps,
                'flow_calculation_time_avg': flow_time_avg,
                'classification_time_avg': class_time_avg,
                'classifications': classifications_json
            }
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        # Return summary
        summary = {
            'camera': camera_idx,
            'frames_processed': frame_count,
            'duration': time.time() - start_time,
            'processing_fps': processing_fps,
            'flow_calculation_time_avg': flow_time_avg,
            'classification_time_avg': class_time_avg
        }
        
        return summary
    
    def benchmark(self):
        """
        Run benchmarks on all test videos with ground truth labels.
        
        Returns:
            Dictionary of benchmark results
        """
        if not self.config['labeled_data']:
            logger.error("No labeled data available for benchmarking")
            return None
        
        # Reset metrics
        self.metrics = {
            'processing_times': [],
            'flow_calculation_times': [],
            'classification_times': [],
            'classifications': [],
            'ground_truth': []
        }
        
        # Run tests on all labeled videos
        results = []
        for video_path, ground_truth in self.config['labeled_data'].items():
            logger.info(f"Benchmarking video: {video_path}, ground truth: {ground_truth}")
            
            # Test video
            result = self.test_video(video_path, ground_truth)
            
            if result:
                results.append(result)
        
        # Calculate overall metrics
        if results:
            overall_fps = np.mean([r['processing_fps'] for r in results])
            overall_flow_time = np.mean([r['flow_calculation_time_avg'] for r in results])
            overall_class_time = np.mean([r['classification_time_avg'] for r in results])
            overall_accuracy = np.mean([r['accuracy'] for r in results if 'accuracy' in r])
            
            # Count classifications by label
            classifications = {}
            for cls in self.metrics['classifications']:
                label = cls['label']
                if label not in classifications:
                    classifications[label] = 0
                classifications[label] += 1
            
            # Count ground truth by label
            ground_truth = {}
            for gt in self.metrics['ground_truth']:
                if gt not in ground_truth:
                    ground_truth[gt] = 0
                ground_truth[gt] += 1
            
            # Save benchmark results
            if self.config['save_results']:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_path = os.path.join(self.config['output_dir'], f"benchmark_{timestamp}.json")
                
                benchmark_results = {
                    'timestamp': timestamp,
                    'overall_fps': overall_fps,
                    'overall_flow_calculation_time': overall_flow_time,
                    'overall_classification_time': overall_class_time,
                    'overall_accuracy': overall_accuracy,
                    'video_results': results,
                    'classifications': classifications,
                    'ground_truth': ground_truth
                }
                
                with open(results_path, 'w') as f:
                    json.dump(benchmark_results, f, indent=2)
            
            # Return summary
            summary = {
                'overall_fps': overall_fps,
                'overall_flow_calculation_time': overall_flow_time,
                'overall_classification_time': overall_class_time,
                'overall_accuracy': overall_accuracy,
                'video_count': len(results),
                'frame_count': sum(r['frames_processed'] for r in results),
                'classification_distribution': classifications
            }
            
            return summary
        
        return None


def main():
    """Main function for running tests."""
    parser = argparse.ArgumentParser(description='Test optical flow-based motion classification')
    
    parser.add_argument('--mode', type=str, default='video',
                        choices=['video', 'camera', 'benchmark'],
                        help='Test mode: video, camera, or benchmark')
    
    parser.add_argument('--input', type=str, default=None,
                        help='Input video file or directory for benchmark mode')
    
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index for camera mode')
    
    parser.add_argument('--duration', type=int, default=60,
                        help='Test duration in seconds for camera mode')
    
    parser.add_argument('--output', type=str, default='test_results',
                        help='Output directory for results')
    
    parser.add_argument('--no-display', action='store_true',
                        help='Disable visualization')
    
    parser.add_argument('--no-save', action='store_true',
                        help='Disable saving results')
    
    parser.add_argument('--label', type=str, default=None,
                        help='Ground truth label for the input video')
    
    args = parser.parse_args()
    
    # Create test configuration
    config = {
        'output_dir': args.output,
        'show_visualization': not args.no_display,
        'save_results': not args.no_save,
        'benchmark_mode': args.mode == 'benchmark'
    }
    
    # Create tester
    tester = OpticalFlowTester(config)
    
    # Run tests based on mode
    if args.mode == 'video':
        if not args.input:
            logger.error("Input video file required for video mode")
            return
        
        logger.info(f"Testing video: {args.input}")
        result = tester.test_video(args.input, args.label)
        
        if result:
            logger.info("Test completed successfully")
            logger.info(f"Processed {result['frames_processed']} frames at {result['processing_fps']:.2f} FPS")
            logger.info(f"Average flow calculation time: {result['flow_calculation_time_avg']*1000:.2f} ms")
            logger.info(f"Average classification time: {result['classification_time_avg']*1000:.2f} ms")
            
            if 'accuracy' in result and result['accuracy'] > 0:
                logger.info(f"Classification accuracy: {result['accuracy']*100:.2f}%")
        else:
            logger.error("Test failed")
    
    elif args.mode == 'camera':
        logger.info(f"Testing camera {args.camera} for {args.duration} seconds")
        result = tester.test_camera(args.camera, args.duration)
        
        if result:
            logger.info("Test completed successfully")
            logger.info(f"Processed {result['frames_processed']} frames at {result['processing_fps']:.2f} FPS")
            logger.info(f"Average flow calculation time: {result['flow_calculation_time_avg']*1000:.2f} ms")
            logger.info(f"Average classification time: {result['classification_time_avg']*1000:.2f} ms")
        else:
            logger.error("Test failed")
    
    elif args.mode == 'benchmark':
        # Load labeled data
        labeled_data = {}
        
        if not args.input:
            logger.error("Input directory required for benchmark mode")
            return
        
        if os.path.isdir(args.input):
            # Look for a labels.json file in the directory
            labels_file = os.path.join(args.input, 'labels.json')
            if os.path.exists(labels_file):
                try:
                    with open(labels_file, 'r') as f:
                        labels = json.load(f)
                        
                        # Convert relative paths to absolute
                        for video_rel_path, label in labels.items():
                            video_path = os.path.join(args.input, video_rel_path)
                            if os.path.exists(video_path):
                                labeled_data[video_path] = label
                except Exception as e:
                    logger.error(f"Error loading labels file: {e}")
            
            # If no labels file or it failed to load, look for videos in subdirectories
            if not labeled_data:
                for category in os.listdir(args.input):
                    category_dir = os.path.join(args.input, category)
                    if os.path.isdir(category_dir):
                        for file in os.listdir(category_dir):
                            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                video_path = os.path.join(category_dir, file)
                                labeled_data[video_path] = category
        else:
            # Single video with label
            if args.label:
                labeled_data[args.input] = args.label
            else:
                logger.error("Label required for benchmarking a single video")
                return
        
        if not labeled_data:
            logger.error("No labeled data found for benchmarking")
            return
        
        # Update tester config with labeled data
        tester.config['labeled_data'] = labeled_data
        
        # Run benchmark
        logger.info(f"Benchmarking {len(labeled_data)} videos")
        result = tester.benchmark()
        
        if result:
            logger.info("Benchmark completed successfully")
            logger.info(f"Overall processing speed: {result['overall_fps']:.2f} FPS")
            logger.info(f"Overall flow calculation time: {result['overall_flow_calculation_time']*1000:.2f} ms")
            logger.info(f"Overall classification time: {result['overall_classification_time']*1000:.2f} ms")
            logger.info(f"Overall accuracy: {result['overall_accuracy']*100:.2f}%")
            logger.info(f"Classification distribution: {result['classification_distribution']}")
        else:
            logger.error("Benchmark failed")


if __name__ == "__main__":
    main()