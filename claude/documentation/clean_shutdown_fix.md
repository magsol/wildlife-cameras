# Clean Shutdown Fix

## Problem

After pressing Ctrl+C, the server process would log shutdown messages but would never actually exit, requiring a manual `kill -9` to terminate:

```
2025-09-23 16:19:17,550 - fastapi_mjpeg_server - INFO - Shutting down camera and server...  
2025-09-23 16:19:17,550 - fastapi_mjpeg_server - INFO - Camera recording stopped
```

Pressing Ctrl+C again would only repeat the same messages without terminating the process.

## Root Cause

The issue was caused by several background threads that were not properly terminated during the shutdown process:

1. **Daemon vs. Non-daemon Threads**: All worker threads were correctly set as daemon threads, but this alone doesn't guarantee clean shutdown in all cases.

2. **Missing Shutdown Flags**: The worker threads were using infinite loops without checking for a shutdown condition, so they would continue running even after the main shutdown sequence started.

3. **No Thread Join**: The main process wasn't waiting for these threads to finish before completing shutdown.

4. **Missing Process Exit**: The process didn't explicitly call `sys.exit()` or `os._exit()` after cleanup.

5. **Multiple Background Thread Types**:
   - `WiFiMonitor._monitor_signal` - Continuously checks WiFi signal strength
   - `TransferManager._transfer_worker` - Processes file transfers
   - `TransferManager._disk_check_worker` - Periodically checks disk usage
   - `MotionEventRecorder._process_events` - Processes motion events

## The Fix

1. **Added a Global Shutdown Event**: Added a global `shutdown_requested` threading event to signal all threads to stop:
   ```python
   # Global shutdown event for signaling threads to terminate
   shutdown_requested = threading.Event()
   ```

2. **Modified Thread Loops**: Updated all worker threads to check this event and exit when it's set:
   ```python
   while not shutdown_requested.is_set():
       # Thread work...
       
   logger.info("Thread exiting")
   ```

3. **Added Sleep Chunking**: For threads with long sleep times, added chunked sleeping to check the shutdown flag more frequently:
   ```python
   # Check disk usage every 5 minutes - use smaller sleeps to check shutdown flag more frequently
   for _ in range(60):  # 60 * 5 seconds = 300 seconds (5 minutes)
       if shutdown_requested.is_set():
           break
       time.sleep(5)
   ```

4. **Created a Shutdown Function**: Added a `shutdown()` function to the motion storage module:
   ```python
   def shutdown():
       """Shutdown all background threads and processes"""
       logger.info("Shutting down motion storage module...")
       
       # Set the shutdown event to signal all threads to exit
       global shutdown_requested
       shutdown_requested.set()
       
       # Wait a bit for threads to notice the shutdown event
       time.sleep(2)
       
       logger.info("Motion storage module shutdown complete")
       
       return True
   ```

5. **Updated Handle_Shutdown**: Modified the FastAPI module's shutdown handler to:
   - Stop the camera recording
   - Signal the streaming shutdown event
   - Call the motion storage module's shutdown function
   - Force exit the process with `os._exit(0)`

6. **Added Uvicorn Timeout**: Set a timeout for uvicorn's graceful shutdown:
   ```python
   uvicorn.run(
       app,
       host=args.host,
       port=args.port,
       log_level="info",
       reload=False,
       timeout_graceful_shutdown=5  # 5 seconds timeout for graceful shutdown
   )
   ```

## Key Insights

1. **Complete Thread Cleanup**: For applications with multiple background threads, it's essential to have a centralized shutdown mechanism that notifies all threads to terminate.

2. **Thread Cooperation**: Background threads should periodically check for a shutdown condition and exit gracefully when requested.

3. **Forced Termination**: In some cases, particularly with external libraries or frameworks, it's necessary to use `os._exit()` to ensure the process terminates completely after cleanup.

4. **Chunked Sleep Times**: For threads with long sleep periods, it's better to use smaller sleep chunks and check for shutdown between them to ensure faster response to shutdown requests.

## Benefits

1. **Clean Termination**: The process now exits cleanly after pressing Ctrl+C, without requiring a force kill.

2. **Resource Cleanup**: All background threads get a chance to clean up resources before termination.

3. **Faster Shutdown**: The process exits within a few seconds of receiving a termination signal.

4. **Better User Experience**: Users no longer need to manually kill the process.