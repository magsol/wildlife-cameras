# User Input Control Flow

```mermaid
flowchart TD
    A[Start Application] --> B{Parse Command Line Arguments}
    B -->|Success| C[Initialize Camera]
    B -->|Failure| D[Display Help and Exit]
    
    C -->|Success| E[Initialize Storage System]
    C -->|Failure| F[Log Error and Exit]
    
    E -->|Success| G[Start FastAPI Server]
    E -->|Failure| H[Log Error and Exit]
    
    G --> I[Wait for User Input]
    
    I --> J[Web Interface Inputs]
    I --> K[API Endpoint Calls]
    I --> L[Command Line Signals]
    
    J --> J1[Stream Request]
    J --> J2[Configuration Update]
    J --> J3[Motion History View]
    J --> J4[Storage Stats View]
    
    J1 --> M[Register Client]
    M -->|Max Clients Not Reached| N[Start Streaming]
    M -->|Max Clients Reached| O[Return Error 503]
    
    J2 --> P[Update Camera Config]
    J2 --> Q[Update Storage Config]
    
    P --> P1[Update Timestamp Settings]
    P --> P2[Update Motion Detection Settings]
    
    Q --> Q1[Update Local Storage Settings]
    Q --> Q2[Update Transfer Settings]
    Q --> Q3[Update WiFi Settings]
    
    K --> K1[Status Endpoint Call]
    K --> K2[Config Endpoint Call]
    K --> K3[Motion Status Endpoint Call]
    K --> K4[Storage Status Endpoint Call]
    K --> K5[Force Transfer Endpoint Call]
    
    K1 --> R[Return Status JSON]
    K2 --> S[Process Config Update]
    K3 --> T[Return Motion Status JSON]
    K4 --> U[Return Storage Stats JSON]
    K5 --> V[Force Transfer of Event]
    
    S --> P
    S --> Q
    
    L --> L1[SIGINT/SIGTERM Signal]
    L --> L2[Other System Signals]
    
    L1 --> W[Start Graceful Shutdown]
    L2 --> X[Handle System Signal]
    
    W --> W1[Stop Camera Recording]
    W1 --> W2[Complete Pending Transfers]
    W2 --> W3[Close Server]
    W3 --> Y[Exit Application]
```

## Detailed Description of User Input Control Flow

### Initial Setup

1. **Start Application**: The application begins execution
2. **Parse Command Line Arguments**: The system parses arguments like resolution, frame rate, motion detection settings, etc.
   - **Success**: Continues with initialization
   - **Failure**: Displays help message and exits

3. **Initialize Camera**: Sets up the camera with configured settings
   - **Success**: Proceeds to initialize storage system
   - **Failure**: Logs error and exits

4. **Initialize Storage System**: Sets up local and remote storage components
   - **Success**: Starts FastAPI server
   - **Failure**: Logs error and exits

5. **Start FastAPI Server**: Initializes web server and endpoints

### User Input Sources

After initialization, the system waits for user input from three main sources:
- **Web Interface Inputs**: Actions taken through the browser UI
- **API Endpoint Calls**: Direct REST API calls
- **Command Line Signals**: SIGINT, SIGTERM, etc.

### Web Interface Inputs

1. **Stream Request**:
   - Checks if maximum client limit is reached
   - If limit not reached, registers client and starts streaming
   - If limit reached, returns 503 error

2. **Configuration Update**:
   - Camera configuration (timestamp, motion detection settings)
   - Storage configuration (local storage, transfer settings, WiFi settings)

3. **Motion History View**:
   - Displays recent motion events with timestamps

4. **Storage Stats View**:
   - Shows storage usage, pending transfers, and network status

### API Endpoint Calls

1. **Status Endpoint** (`/status`):
   - Returns camera and server status information

2. **Config Endpoint** (`/config`):
   - Processes configuration updates
   - Updates camera or storage settings

3. **Motion Status Endpoint** (`/motion_status`):
   - Returns current motion detection status and history

4. **Storage Status Endpoint** (`/storage/status`):
   - Returns storage usage statistics and pending transfers

5. **Force Transfer Endpoint** (`/storage/transfer/{event_id}`):
   - Forces immediate transfer of a specific motion event

### Command Line Signals

1. **SIGINT/SIGTERM**:
   - Triggers graceful shutdown sequence:
     - Stops camera recording
     - Completes pending transfers if possible
     - Closes server connections
     - Exits application

2. **Other System Signals**:
   - Handled according to system requirements

### Configuration Updates

1. **Camera Configuration**:
   - **Timestamp Settings**: Show/hide timestamp, position, color, size
   - **Motion Detection Settings**: Enable/disable, sensitivity, minimum area, highlighting

2. **Storage Configuration**:
   - **Local Storage Settings**: Path, maximum size
   - **Transfer Settings**: Enable/disable, throttle, scheduling
   - **WiFi Settings**: Signal monitoring, adaptive throttling

### Shutdown Sequence

1. **Stop Camera Recording**: Safely stops camera recording
2. **Complete Pending Transfers**: Attempts to finish active transfers
3. **Close Server**: Closes all connections and shuts down server
4. **Exit Application**: Terminates the application