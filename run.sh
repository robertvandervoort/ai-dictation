#!/bin/bash

echo "== AI Dictation Launcher (Unix/Linux/Mac) =="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found. Please install Python 3.8 or newer."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Detect OS type
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS_TYPE="linux"
    echo "Detected Linux OS"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
    echo "Detected macOS"
else
    OS_TYPE="other"
    echo "Detected other Unix OS: $OSTYPE"
fi

# Set up virtual environment path
VENV_DIR="venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        exit 1
    fi
    echo "Virtual environment created successfully."
fi

# Activate venv
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Update pip to latest version FIRST
echo "Updating pip to the latest version..."
python3 -m pip install --upgrade pip
if [ $? -ne 0 ]; then
    echo "Warning: Failed to update pip. Continuing with existing version."
fi

# Install build tools that might be needed SECOND
echo "Installing build tools..."
pip install setuptools wheel cython
if [ $? -ne 0 ]; then
    echo "Warning: Failed to install build tools. Some packages might fail to install."
fi

# Install PyTorch with appropriate version
if [ "$OS_TYPE" == "linux" ]; then
    echo "Installing PyTorch with CUDA 12.1 support for Linux..."
    pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install PyTorch with CUDA support. Falling back to CPU version."
        pip install torch==2.1.2 torchaudio==2.1.2
    else
        echo "PyTorch with CUDA support installed successfully."
    fi
elif [ "$OS_TYPE" == "macos" ]; then
    echo "Installing PyTorch for macOS..."
    pip install torch==2.1.2 torchaudio==2.1.2
    echo "PyTorch installed successfully."
else
    echo "Installing PyTorch for generic Unix..."
    pip install torch==2.1.2 torchaudio==2.1.2
    echo "PyTorch installed successfully."
fi

# Install basic requirements
echo "Installing basic requirements..."
pip install streamlit==1.32.0 numpy==1.26.3 librosa==0.10.1 soundfile==0.12.1 faster-whisper==0.10.0 ffmpeg-python==0.2.0 openai-whisper==20231117 protobuf==4.25.1
if [ $? -ne 0 ]; then
    echo "Error: Failed to install basic requirements."
    exit 1
fi
echo "Basic requirements installed successfully."

# Install appropriate speech recognition libraries based on OS
if [ "$OS_TYPE" == "linux" ]; then
    # For Linux: Use seamless_communication directly
    echo "Installing seamless_communication for Linux..."
    
    # Check for additional dependencies specific to Linux
    if ! command -v ffmpeg &> /dev/null; then
        echo "Warning: ffmpeg not found. Please install it for full functionality."
        echo "  Ubuntu/Debian: sudo apt install ffmpeg"
        echo "  CentOS/RHEL: sudo yum install ffmpeg"
    fi
    
    # Install pyannote.core first (required by pyannote.audio)
    echo "Installing pyannote.core..."
    pip install pyannote.core
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pyannote.core. Speaker diarization will not work."
        exit 1
    fi
    
    # Install pyannote.audio for Linux
    echo "Installing pyannote.audio..."
    pip install pyannote.audio==3.1.1
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install pyannote.audio. Trying alternative approach..."
        pip install pyannote.audio==3.1.1 --no-build-isolation
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install pyannote.audio. Speaker diarization may not work."
        fi
    fi
    
    # Using specific versions from documentation
    pip install fairseq2==0.2.0 sentencepiece==0.1.99
    
    # On Linux, try to install fairseq2n (native component)
    pip install fairseq2n==0.2.0
    
    # Install seamless_communication from PyPI if available, otherwise from GitHub
    pip install seamless_communication==1.0.0 2>/dev/null || pip install git+https://github.com/facebookresearch/seamless_communication.git@v1.0.0
    
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install seamless_communication. Some features may not be available."
    else
        echo "seamless_communication installed successfully."
    fi
else
    # For macOS and other Unix: Use transformers-based implementation
    echo "Installing Transformers with Seamless M4T support..."
    pip install transformers==4.38.2 sentencepiece==0.1.99 accelerate==0.26.1
    
    # Install pyannote.core first (required by pyannote.audio)
    echo "Installing pyannote.core..."
    pip install pyannote.core
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install pyannote.core. Speaker diarization will not work."
        exit 1
    fi
    
    # Install pyannote.audio for macOS
    echo "Installing pyannote.audio for macOS..."
    pip install pyannote.audio==3.1.1
    if [ $? -ne 0 ]; then
        echo "Warning: Failed to install pyannote.audio. Trying alternative approach..."
        pip install pyannote.audio==3.1.1 --no-build-isolation
        if [ $? -ne 0 ]; then
            echo "Error: Failed to install pyannote.audio. Speaker diarization may not work."
        fi
    fi
    
    # Check if ffmpeg is available for audio processing
    if ! command -v ffmpeg &> /dev/null; then
        echo "Warning: ffmpeg not found. Some audio processing functionality may be limited."
        if [ "$OS_TYPE" == "macos" ]; then
            echo "  Install with: brew install ffmpeg"
        else
            echo "  Please install ffmpeg using your package manager"
        fi
    else
        echo "ffmpeg is available."
    fi
fi

# Run the application
echo "Starting AI Dictation application..."
streamlit run dictate.py

# This line will only be reached if streamlit is stopped
deactivate

exit 0 