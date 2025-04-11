import streamlit as st
import torch
import numpy as np
import pyannote.audio as paa
#from pyannote.core import Segment
import tempfile
import os
import librosa
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import traceback
import time
import uuid
import warnings
import platform
import hashlib
import psutil
import importlib
import types
# import sys
import logging

#if you have / need to use a huggingface token, set it here
HF_TOKEN = ""

#keep streamlit from whining about torch.classes    
torch.classes.__path__ = []

# Set to True to enable regular (non-debug) console output
# Set to False to suppress most console output
DEBUG_CONSOLE_OUTPUT = False

# Configure logging to suppress noisy messages
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("pyannote").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("librosa").setLevel(logging.WARNING)
logging.getLogger("streamlit").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions now support par-call bakcend dispatch")
warnings.filterwarnings("ignore", message="torch.classes is an experimental")
warnings.filterwarnings("ignore", message="torch._C._get_custom_class_python_wrapper")
warnings.filterwarnings("ignore", message="Tried to instantiate class")

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    if DEBUG_CONSOLE_OUTPUT:
        print("pynvml not found. Using PyTorch metrics for GPU memory usage.")

# Check for Apple Silicon (Metal) support
APPLE_SILICON_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False

# Check for AMD ROCm support
try:
    import torch.utils.hipify
    AMD_ROCM_AVAILABLE = hasattr(torch.version, 'hip') and torch.version.hip is not None
except (ImportError, AttributeError):
    AMD_ROCM_AVAILABLE = False

# Add a comprehensive language map for Whisper at the top of the file, after the imports
# This defines all languages that Whisper supports for both transcription and translation
WHISPER_LANGUAGE_MAP = {
    "en": "English",
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "ko": "Korean",
    "fr": "French",
    "ja": "Japanese",
    "pt": "Portuguese",
    "tr": "Turkish",
    "pl": "Polish",
    "ca": "Catalan",
    "nl": "Dutch",
    "ar": "Arabic",
    "sv": "Swedish",
    "it": "Italian",
    "id": "Indonesian",
    "hi": "Hindi",
    "fi": "Finnish",
    "vi": "Vietnamese",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "el": "Greek",
    "ms": "Malay",
    "cs": "Czech",
    "ro": "Romanian",
    "da": "Danish",
    "hu": "Hungarian",
    "ta": "Tamil",
    "no": "Norwegian",
    "th": "Thai",
    "ur": "Urdu",
    "hr": "Croatian",
    "bg": "Bulgarian",
    "lt": "Lithuanian",
    "la": "Latin",
    "mi": "Maori",
    "ml": "Malayalam",
    "cy": "Welsh",
    "sk": "Slovak",
    "te": "Telugu",
    "fa": "Persian",
    "lv": "Latvian",
    "bn": "Bengali",
    "sr": "Serbian",
    "az": "Azerbaijani",
    "sl": "Slovenian",
    "kn": "Kannada",
    "et": "Estonian",
    "mk": "Macedonian",
    "br": "Breton",
    "eu": "Basque",
    "is": "Icelandic",
    "hy": "Armenian",
    "ne": "Nepali",
    "mn": "Mongolian",
    "bs": "Bosnian",
    "kk": "Kazakh",
    "sq": "Albanian",
    "sw": "Swahili",
    "gl": "Galician",
    "mr": "Marathi",
    "pa": "Punjabi",
    "si": "Sinhala",
    "km": "Khmer",
    "sn": "Shona",
    "yo": "Yoruba",
    "so": "Somali",
    "af": "Afrikaans",
    "oc": "Occitan",
    "ka": "Georgian",
    "be": "Belarusian",
    "tg": "Tajik",
    "sd": "Sindhi",
    "gu": "Gujarati",
    "am": "Amharic",
    "yi": "Yiddish",
    "lo": "Lao",
    "uz": "Uzbek",
    "fo": "Faroese",
    "ht": "Haitian Creole",
    "ps": "Pashto",
    "tk": "Turkmen",
    "nn": "Nynorsk",
    "mt": "Maltese",
    "sa": "Sanskrit",
    "lb": "Luxembourgish",
    "my": "Myanmar",
    "bo": "Tibetan",
    "tl": "Tagalog",
    "mg": "Malagasy",
    "as": "Assamese",
    "tt": "Tatar",
    "haw": "Hawaiian",
    "ln": "Lingala",
    "ha": "Hausa",
    "ba": "Bashkir",
    "jw": "Javanese",
    "su": "Sundanese",
}

# Helper functions to convert between language names and codes
def get_language_code(language_name):
    """Convert a language name to its code"""
    for code, name in WHISPER_LANGUAGE_MAP.items():
        if name == language_name:
            return code
    return None

def get_language_name(language_code):
    """Convert a language code to its name"""
    return WHISPER_LANGUAGE_MAP.get(language_code, language_code)

# Add a debug logging function right after the imports and global variables
def debug_log(message):
    """Log debug information only when debug mode is enabled."""
    if 'debug_mode' in st.session_state and st.session_state.debug_mode:
        print(message)

# Function to get memory usage metrics
def get_system_memory_metrics(source="psutil", model=None):
    """Get system RAM usage metrics in a consistent format."""
    memory = psutil.virtual_memory()
    metrics = {
        "used": memory.used / (1024**3),  # GB
        "total": memory.total / (1024**3),  # GB
        "free": memory.available / (1024**3),  # GB
        "usage_percent": memory.percent,
        "source": source
    }
    
    # Add model info if provided
    if model:
        metrics["model"] = model
        
    return metrics

def get_memory_metrics(use_gpu=False, gpu_type=None):
    """Get system RAM or GPU memory usage."""
    if not use_gpu:
        # System RAM usage
        return get_system_memory_metrics()
    
    # NVIDIA GPU
    if gpu_type == "nvidia" and torch.cuda.is_available():
        # Try using direct nvidia-smi call first (most accurate)
        try:
            import subprocess
            
            # Run nvidia-smi command to get memory info
            sp = subprocess.Popen(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            out_str = sp.communicate()
            out_list = out_str[0].decode("utf-8").strip().split(',')
            
            if len(out_list) >= 3:
                # Convert values from MiB to GB (1024 for binary conversion)
                total_gpu_mem = float(out_list[0].strip()) / 1024
                used_gpu_mem = float(out_list[1].strip()) / 1024
                free_gpu_mem = float(out_list[2].strip()) / 1024
                
                # Calculate percentage
                usage_percent = (used_gpu_mem / total_gpu_mem) * 100 if total_gpu_mem > 0 else 0
                
                # Print nvidia-smi values for debugging
                debug_log("GPU Memory Debugging (nvidia-smi):")
                debug_log(f"  nvidia-smi total: {total_gpu_mem:.2f} GB")
                debug_log(f"  nvidia-smi used: {used_gpu_mem:.2f} GB")
                debug_log(f"  nvidia-smi free: {free_gpu_mem:.2f} GB")
                debug_log(f"  FINAL VALUES (nvidia-smi): used={used_gpu_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
                
                return {
                    "used": used_gpu_mem,
                    "total": total_gpu_mem,
                    "free": free_gpu_mem,
                    "usage_percent": usage_percent,
                    "source": "nvidia-smi"
                }
            
        except Exception as e:
            debug_log(f"Error getting GPU metrics from nvidia-smi: {e}")
            # Fall back to other methods
        
        # Fall back to NVML if nvidia-smi failed
        if PYNVML_AVAILABLE:
            try:
                # Initialize NVML
                pynvml.nvmlInit()
                
                # Get the device handle for the first GPU (index 0)
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Convert to GB
                total_gpu_mem = mem_info.total / (1024**3)
                used_gpu_mem = mem_info.used / (1024**3)
                free_gpu_mem = mem_info.free / (1024**3)
                
                # Print NVML values for debugging
                debug_log("GPU Memory Debugging (NVML):")
                debug_log(f"  NVML total: {total_gpu_mem:.2f} GB")
                debug_log(f"  NVML used: {used_gpu_mem:.2f} GB")
                debug_log(f"  NVML free: {free_gpu_mem:.2f} GB")
                
                # Calculate percentage
                usage_percent = (used_gpu_mem / total_gpu_mem) * 100
                
                # Shut down NVML
                pynvml.nvmlShutdown()
                
                debug_log(f"  FINAL VALUES (NVML): used={used_gpu_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
                
                return {
                    "used": used_gpu_mem,
                    "total": total_gpu_mem,
                    "free": free_gpu_mem,
                    "usage_percent": usage_percent,
                    "source": "nvml"
                }
            except Exception as e:
                debug_log(f"Error getting GPU metrics from NVML: {e}")
                # Fall back to PyTorch metrics
                pass
                
        # Final fallback to PyTorch metrics if all else failed
        debug_log("GPU Memory Debugging (PyTorch fallback):")
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        reserved_gpu_mem = torch.cuda.memory_reserved(0) / (1024**3)  # GB
        allocated_gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        free_gpu_mem = total_gpu_mem - reserved_gpu_mem
        
        # Final values to return
        used_mem = reserved_gpu_mem  # Use reserved memory with NO padding
        usage_percent = (used_mem / total_gpu_mem) * 100 if total_gpu_mem > 0 else 0
        
        debug_log(f"  PyTorch total: {total_gpu_mem:.2f} GB")
        debug_log(f"  PyTorch reserved: {reserved_gpu_mem:.2f} GB")
        debug_log(f"  PyTorch allocated: {allocated_gpu_mem:.2f} GB")
        debug_log(f"  FINAL VALUES (PyTorch): used={used_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
        
        return {
            "used": used_mem,  
            "allocated": allocated_gpu_mem,  
            "total": total_gpu_mem,
            "free": free_gpu_mem,
            "usage_percent": usage_percent,
            "source": "pytorch"
        }
    
    # Apple Silicon (M series)
    elif gpu_type == "apple" and APPLE_SILICON_AVAILABLE:
        try:
            # Unfortunately, PyTorch doesn't provide memory info for MPS
            # Use subprocess to run system command to get memory info
            import subprocess
            
            # Try to get Apple GPU information using sysctl
            if platform.system() == "Darwin":
                try:
                    # Get total GPU memory (this is an approximation)
                    # This command gets integrated GPU memory on Apple Silicon
                    # Note: The actual available GPU memory might be dynamic and share with system memory
                    result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse the memory size (total system memory is used as an approximation)
                        total_mem_bytes = int(result.stdout.strip().split(':')[1].strip())
                        total_gpu_mem = total_mem_bytes / (1024**3) / 2  # Assume half of system memory
                        
                        # Get Apple Silicon model info
                        model_result = subprocess.run(['sysctl', 'hw.model'], capture_output=True, text=True)
                        model_name = "Apple Silicon"
                        if model_result.returncode == 0:
                            model_name = model_result.stdout.strip().split(':')[1].strip()
                        
                        # Get active memory usage - this is an approximation
                        # For M series, we don't have exact GPU memory usage, so estimate based on allocated tensor memory
                        if hasattr(torch.mps, 'current_allocated_memory'):
                            allocated_bytes = torch.mps.current_allocated_memory()
                            used_gpu_mem = allocated_bytes / (1024**3)
                        else:
                            # Assume 25% usage if we can't get actual numbers
                            used_gpu_mem = total_gpu_mem * 0.25
                        
                        free_gpu_mem = total_gpu_mem - used_gpu_mem
                        usage_percent = (used_gpu_mem / total_gpu_mem) * 100
                        
                        debug_log("GPU Memory Debugging (Apple Silicon):")
                        debug_log(f"  Apple model: {model_name}")
                        debug_log(f"  Approximate total: {total_gpu_mem:.2f} GB")
                        debug_log(f"  Estimated used: {used_gpu_mem:.2f} GB")
                        debug_log(f"  Estimated free: {free_gpu_mem:.2f} GB")
                        debug_log(f"  FINAL VALUES (Apple Silicon): used={used_gpu_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
                        
                        return {
                            "used": used_gpu_mem,
                            "total": total_gpu_mem,
                            "free": free_gpu_mem,
                            "usage_percent": usage_percent,
                            "source": "apple-sysctl",
                            "model": model_name
                        }
                except Exception as e:
                    debug_log(f"Error getting Apple Silicon metrics: {e}")
                    # Fall back to a simple estimate
                    pass
                    
            # Fallback: Return estimated values
            # For Apple Silicon, memory is dynamically shared with system memory,
            # so return a simple approximation
            system_memory = psutil.virtual_memory()
            total_gpu_mem = system_memory.total / (1024**3) / 2  # Assume half of system memory is available for GPU
            used_gpu_mem = total_gpu_mem * 0.25  # Just a placeholder
            free_gpu_mem = total_gpu_mem - used_gpu_mem
            usage_percent = 25.0  # Placeholder
            
            return {
                "used": used_gpu_mem,
                "total": total_gpu_mem,
                "free": free_gpu_mem,
                "usage_percent": usage_percent,
                "source": "estimated",
                "model": "Apple Silicon"
            }
        except Exception as e:
            debug_log(f"Error getting Apple Silicon metrics: {e}")
            # Fall back to system memory info
            return get_system_memory_metrics(source="system-ram-fallback", model="Apple Silicon")
    
    # AMD GPU (ROCm)
    elif gpu_type == "amd" and AMD_ROCM_AVAILABLE:
        try:
            # Use rocm-smi if available
            import subprocess
            
            try:
                # Run rocm-smi command to get memory info
                rocm_process = subprocess.Popen(
                    ['rocm-smi', '--showmeminfo', 'vram', '--json'], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                out_str, _ = rocm_process.communicate()
                
                # Parse JSON output
                import json
                data = json.loads(out_str)
                
                # Extract memory info from first GPU
                card_data = next(iter(data.values()))
                total_gpu_mem = int(card_data.get("VRAM Total", "0").split()[0]) / 1024  # Convert to GB
                used_gpu_mem = int(card_data.get("VRAM Used", "0").split()[0]) / 1024
                free_gpu_mem = total_gpu_mem - used_gpu_mem
                usage_percent = (used_gpu_mem / total_gpu_mem) * 100 if total_gpu_mem > 0 else 0
                
                # Get GPU name
                name_process = subprocess.Popen(
                    ['rocm-smi', '--showproductname'], 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                name_out, _ = name_process.communicate()
                gpu_name = name_out.decode('utf-8').strip().split('\n')[-1].strip()
                
                debug_log("GPU Memory Debugging (ROCm):")
                debug_log(f"  AMD GPU: {gpu_name}")
                debug_log(f"  rocm-smi total: {total_gpu_mem:.2f} GB")
                debug_log(f"  rocm-smi used: {used_gpu_mem:.2f} GB")
                debug_log(f"  rocm-smi free: {free_gpu_mem:.2f} GB")
                debug_log(f"  FINAL VALUES (rocm-smi): used={used_gpu_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
                
                return {
                    "used": used_gpu_mem,
                    "total": total_gpu_mem,
                    "free": free_gpu_mem,
                    "usage_percent": usage_percent,
                    "source": "rocm-smi",
                    "model": gpu_name
                }
            except Exception as e:
                debug_log(f"Error getting AMD GPU metrics from rocm-smi: {e}")
                # Fall back to PyTorch metrics
                pass
            
            # Fall back to PyTorch metrics for AMD GPU
            # HIP (ROCm) should provide memory stats similar to CUDA
            if hasattr(torch.hip, 'get_device_properties'):
                total_gpu_mem = torch.hip.get_device_properties(0).total_memory / (1024**3)
                allocated_gpu_mem = torch.hip.memory_allocated(0) / (1024**3)
                reserved_gpu_mem = torch.hip.memory_reserved(0) / (1024**3)
                free_gpu_mem = total_gpu_mem - reserved_gpu_mem
                usage_percent = (reserved_gpu_mem / total_gpu_mem) * 100 if total_gpu_mem > 0 else 0
                
                debug_log("GPU Memory Debugging (PyTorch HIP):")
                debug_log(f"  PyTorch total: {total_gpu_mem:.2f} GB")
                debug_log(f"  PyTorch reserved: {reserved_gpu_mem:.2f} GB")
                debug_log(f"  PyTorch allocated: {allocated_gpu_mem:.2f} GB")
                debug_log(f"  FINAL VALUES (PyTorch HIP): used={reserved_gpu_mem:.2f}GB, total={total_gpu_mem:.2f}GB, percent={usage_percent:.2f}%")
                
                return {
                    "used": reserved_gpu_mem,
                    "allocated": allocated_gpu_mem,
                    "total": total_gpu_mem,
                    "free": free_gpu_mem,
                    "usage_percent": usage_percent,
                    "source": "pytorch-hip"
                }
        except Exception as e:
            debug_log(f"Error getting AMD GPU metrics: {e}")
            # Fall back to system memory metrics
            return get_system_memory_metrics(source="system-ram-fallback", model="AMD GPU")
    
    # If no GPU or unsupported GPU type, return system RAM usage
    return get_system_memory_metrics()

# Initialize session state variables if they don't exist
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'speaker_names' not in st.session_state:
    st.session_state.speaker_names = {}
if 'audio_samples' not in st.session_state:
    st.session_state.audio_samples = {}
if 'transcript_data' not in st.session_state:
    st.session_state.transcript_data = None
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None
if 'preferred_language' not in st.session_state:
    st.session_state.preferred_language = "English"
if 'translated_segments' not in st.session_state:
    st.session_state.translated_segments = {}
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'last_memory_update' not in st.session_state:
    st.session_state.last_memory_update = 0
if 'memory_needs_update' not in st.session_state:
    st.session_state.memory_needs_update = False

# Global variables
# transformers_debug = False

# Display CUDA information - helpful for debugging
gpu_info = {}

# Check for different GPU types
cuda_available = torch.cuda.is_available()
mps_available = APPLE_SILICON_AVAILABLE
rocm_available = AMD_ROCM_AVAILABLE

if cuda_available:
    device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
    cuda_version = torch.version.cuda if hasattr(torch.version, 'cuda') else "Unknown"
    debug_log(f"CUDA is available: {cuda_available}")
    debug_log(f"CUDA version: {cuda_version}")
    debug_log(f"GPU count: {device_count}")
    debug_log(f"GPU device name: {device_name}")
    gpu_info = {
        "type": "nvidia",
        "name": device_name,
        "count": device_count,
        "version": cuda_version
    }
elif mps_available:
    debug_log("MPS (Apple Silicon) is available for GPU acceleration")
    # Get Apple Silicon model info if possible
    model_name = "Apple Silicon"
    try:
        import subprocess
        model_result = subprocess.run(['sysctl', 'hw.model'], capture_output=True, text=True)
        if model_result.returncode == 0:
            model_name = model_result.stdout.strip().split(':')[1].strip()
    except:
        pass
    
    debug_log(f"Apple Silicon model: {model_name}")
    gpu_info = {
        "type": "apple",
        "name": model_name,
        "count": 1,
        "version": platform.mac_ver()[0] if platform.system() == "Darwin" else "Unknown"
    }
elif rocm_available:
    debug_log("ROCm is available for AMD GPU acceleration")
    hip_version = torch.version.hip if hasattr(torch.version, 'hip') else "Unknown"
    
    # Try to get AMD GPU name using rocm-smi
    gpu_name = "AMD GPU"
    try:
        import subprocess
        name_process = subprocess.Popen(
            ['rocm-smi', '--showproductname'], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        name_out, _ = name_process.communicate()
        gpu_name = name_out.decode('utf-8').strip().split('\n')[-1].strip()
    except:
        pass
    
    debug_log(f"ROCm version: {hip_version}")
    debug_log(f"AMD GPU name: {gpu_name}")
    gpu_info = {
        "type": "amd",
        "name": gpu_name,
        "count": 1,
        "version": hip_version
    }
else:
    debug_log("No GPU acceleration is available. Running on CPU only.")
    gpu_info = {
        "type": "cpu",
        "name": "CPU",
        "count": 0,
        "version": "N/A"
    }

# Fix SpeechBrain deprecation warnings
os.environ["SPEECHBRAIN_SILENCE_DEPRECATION_PRETRAINED"] = "1"

# Fix PyAnnote std warning by setting a numeric computation environment variable
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::RuntimeWarning"

# For HuggingFace cache settings on Windows - properly handle without ignoring
if os.name == 'nt':  # Windows
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "proper_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = cache_dir

# Fix for Streamlit's torch.classes inspection errors
def patch_streamlit_file_watcher():
    """Patch Streamlit's file watcher to handle torch.classes correctly."""
    try:
        import types
        import importlib
        from streamlit.watcher import local_sources_watcher
        
        # Get original implementation to patch
        original_extract_paths = local_sources_watcher._extract_module_paths
        original_get_module_paths = local_sources_watcher.get_module_paths
        
        # Make a fully comprehensive patch that excludes all torch modules from inspection
        def is_torch_module(module):
            """Check if a module is a PyTorch module that should be excluded."""
            if not hasattr(module, "__name__"):
                return False
                
            name = module.__name__
            return (name.startswith("torch") or 
                    "_torch" in name or 
                    "torch_" in name or 
                    "torchaudio" in name)
        
        # Create patched version for module paths extraction
        def patched_extract_paths(module):
            """Safe extraction of module paths that won't crash on torch modules."""
            try:
                # Skip any PyTorch module completely
                if is_torch_module(module):
                    return []
                
                # For other modules, use the original function
                return original_extract_paths(module)
            except Exception:
                # For any other module that causes errors, return empty list
                return []
        
        # Patch get_module_paths to avoid processing torch modules
        def patched_get_module_paths(module):
            """Safely get module paths."""
            try:
                # Skip any PyTorch module completely
                if is_torch_module(module):
                    return []
                    
                # Handle the lambda m: list(m.__path__._path) case specifically
                # This is the main source of errors with torch modules
                if hasattr(module, "__name__") and not hasattr(module, "__path__"):
                    return []
                
                return original_get_module_paths(module)
            except Exception:
                # For any other module that causes errors, return empty list
                return []
        
        # Create a safety wrapper for module.__path__ access
        class SafePathWrapper:
            def __init__(self, module):
                self.module = module
                
            @property
            def __path__(self):
                # Return a dummy path property with a safe _path attribute
                if hasattr(self.module, "__path__") and hasattr(self.module.__path__, "_path"):
                    return self.module.__path__
                else:
                    # Create a dummy object with a _path that won't crash when listed
                    dummy = types.SimpleNamespace()
                    dummy._path = []
                    return dummy
        
        # Patch the specific pattern that causes the __path__._path access
        original_import_module = importlib.import_module
        
        def safe_import_module(name, package=None):
            """Wrapper for import_module that protects torch modules."""
            module = original_import_module(name, package)
            if is_torch_module(module):
                # Return a wrapped module that won't crash on __path__._path access
                return SafePathWrapper(module)
            return module
        
        # Apply the patches
        local_sources_watcher._extract_module_paths = patched_extract_paths
        local_sources_watcher.get_module_paths = patched_get_module_paths
        importlib.import_module = safe_import_module
        
        # For additional safety, directly modify the lambda in the local_sources_watcher
        # This is a bit of a hack, but it targets the exact source of the error
        if hasattr(local_sources_watcher, "_get_paths_from_module_spec"):
            # The function that contains the problematic lambda - patch each handler
            original_get_paths = local_sources_watcher._get_paths_from_module_spec
            
            def patched_get_paths(spec):
                """A safer version of _get_paths_from_module_spec."""
                # If it's a torch module, return an empty list
                if spec and hasattr(spec, "name") and (
                    spec.name.startswith("torch") or 
                    "_torch" in spec.name or 
                    "torch_" in spec.name or 
                    "torchaudio" in spec.name
                ):
                    return []
                
                # Otherwise use the original function
                return original_get_paths(spec)
            
            # Apply this patch too
            local_sources_watcher._get_paths_from_module_spec = patched_get_paths
            
    except Exception as e:
        if DEBUG_CONSOLE_OUTPUT:
            print(f"Warning: Advanced Streamlit patch failed: {e}")
        # Try simple patch as fallback
        try:
            from streamlit.watcher import local_sources_watcher
            
            # Simple blacklist approach as fallback
            def simple_extract_paths(module):
                if (hasattr(module, "__name__") and 
                    ("torch" in module.__name__ or "torchaudio" in module.__name__)):
                    return []
                try:
                    if hasattr(local_sources_watcher, "_extract_module_paths"):
                        original = local_sources_watcher._extract_module_paths
                        return original(module)
                    else:
                        return []
                except:
                    return []
            
            # Apply simple patch
            if hasattr(local_sources_watcher, "_extract_module_paths"):
                local_sources_watcher._extract_module_paths = simple_extract_paths
                
        except Exception as e2:
            if DEBUG_CONSOLE_OUTPUT:
                print(f"Warning: Streamlit simple patch also failed: {e2}")

# Apply the Streamlit patch
patch_streamlit_file_watcher()

def safe_remove_file(file_path, max_retries=3, retry_delay=0.5):
    """Safely remove a file with retries, suppressing errors if the file can't be deleted."""
    for attempt in range(max_retries):
        try:
            os.unlink(file_path)
            return True  # Successfully removed
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait before retrying
            else:
                # On last attempt, just log it and continue
                if os.path.exists(file_path):
                    if DEBUG_CONSOLE_OUTPUT or st.session_state.get('debug_mode', False):
                        print(f"Warning: Could not remove temporary file: {file_path}")
                return False
        except FileNotFoundError:
            # File already gone
            return True
        except Exception as e:
            if DEBUG_CONSOLE_OUTPUT or st.session_state.get('debug_mode', False):
                print(f"Error removing file {file_path}: {e}")
            return False

def torch_parameter_hash(parameter):
    return parameter.data.numpy().tobytes()

def get_device_for_model(use_gpu=False):
    """Helper function to get the appropriate device for loading models"""
    if not use_gpu:
        return "cpu"
    
    if cuda_available:
        return "cuda"
    elif mps_available:
        return "mps"
    elif rocm_available:
        return "cuda"  # PyTorch ROCm uses "cuda" as device name for compatibility
    else:
        return "cpu"

# Model Selection with Hugging Face Authentication
@st.cache_resource(hash_funcs={torch.nn.parameter.Parameter: torch_parameter_hash})
def load_diarization_model(use_gpu=False):
    '''Load the speaker diarization model'''
    try:
        hf_token = "HF_TOKEN"
        pipeline = paa.Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        
        # Move to appropriate device if requested and available
        if use_gpu:
            device = get_device_for_model(use_gpu)
            pipeline.to(torch.device(device))
            
        return pipeline
    except Exception as e:
        st.error(f"Error loading diarization model: {e}")
        raise

@st.cache_resource
def load_whisper_model(model_size="base", use_gpu=False):
    '''Load the Whisper model using faster-whisper (CTranslate2 implementation)'''
    try:
        # Set device and compute type based on GPU availability
        device = "cpu"
        compute_type = "int8"
        
        if use_gpu:
            # CTranslate2 supports CUDA and CPU, but not MPS directly
            # For Apple Silicon, we use CPU with optimized int8 computation
            # For NVIDIA and AMD with CUDA/ROCm, we use GPU
            if cuda_available or rocm_available:
                device = "cuda"
                compute_type = "float16"
            else:
                # On Apple Silicon, use optimized CPU mode (or auto which may use Metal via system libraries)
                device = "auto" if platform.system() == "Darwin" else "cpu"
        
        # Use GPU if requested and available
        model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="./models")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        raise

@st.cache_resource
def load_wav2vec2_model(model_name="facebook/wav2vec2-base-960h", use_gpu=False):
    '''Load the Wav2Vec2 ASR model and processor'''
    try:
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Move to appropriate device if requested and available
        if use_gpu:
            device = get_device_for_model(use_gpu)
            model = model.to(device)
            
        return model, processor
    except Exception as e:
        st.error(f"Error loading Wav2Vec2 model: {e}")
        raise

def transcribe_whisper_segment(audio_path, start_time, end_time, model, sample_rate=16000, language=None, task="transcribe"):
    """Transcribe or translate a specific audio segment using Whisper"""
    try:
        # Use librosa for more reliable segment extraction
        audio, sr = librosa.load(audio_path, sr=sample_rate, offset=start_time, duration=(end_time-start_time))
        
        # If segment is too short, return empty string
        if len(audio) < 0.5 * sample_rate:  # Less than 500ms
            return "", None
        
        # Create a uniquely named temporary file
        temp_filename = f"whisper_segment_{uuid.uuid4().hex}.wav"
        temp_filepath = os.path.join(tempfile.gettempdir(), temp_filename)
        
        try:
            # Save the segment to a temporary file
            sf.write(temp_filepath, audio, sample_rate)
            
            # Run the inference with explicit language if provided
            if language and language.lower() != "auto-detect":
                segments, info = model.transcribe(temp_filepath, beam_size=5, language=language.lower(), task=task)
            else:
                segments, info = model.transcribe(temp_filepath, beam_size=5, task=task)
            
            # Get detected language from info
            detected_language = None
            
            # Add debug output to console
            if hasattr(info, 'language') and info.language:
                debug_log(f"Debug - TranscriptionInfo object found with language: {info.language}, Probability: {info.language_probability:.2f}")
                
                # Check all available attributes of info
                debug_log("Debug - TranscriptionInfo object attributes:" + str(dir(info)))
                
                # Verify access to language attribute directly
                try:
                    language_attr = getattr(info, 'language')
                    debug_log(f"Debug - Direct attribute access: info.language = '{language_attr}'")
                except Exception as e:
                    debug_log(f"Debug - Error accessing language attribute: {e}")
                
                # Try to get the language using __dict__ if available
                if hasattr(info, '__dict__'):
                    debug_log(f"Debug - info.__dict__ = {info.__dict__}")
                
                # Try alternative methods to get language
                try:
                    language_str = str(info)
                    debug_log(f"Debug - info as string: {language_str}")
                except Exception as e:
                    debug_log(f"Debug - Error converting info to string: {e}")
            else:
                debug_log("Debug - info object doesn't have language attribute or is None")
            
            # Handle info whether it's a dictionary or TranscriptionInfo object
            try:
                # Try to get language as attribute (TranscriptionInfo object in newer versions)
                if hasattr(info, 'language') and info.language:
                    lang_code = info.language
                    debug_log(f"Debug - Got lang_code='{lang_code}' from info.language attribute")
                # Try to get language as dictionary key (older versions)
                elif isinstance(info, dict) and "language" in info and info["language"]:
                    lang_code = info["language"]
                    debug_log(f"Debug - Got lang_code='{lang_code}' from info dictionary")
                else:
                    lang_code = None
                    debug_log("Debug - No language code found in info object")
                
                # Map language code to full name if we have a language code
                if lang_code:
                    detected_language = WHISPER_LANGUAGE_MAP.get(lang_code, lang_code.title())
                    debug_log(f"Debug - Mapped language '{lang_code}' to '{detected_language}' from map: {WHISPER_LANGUAGE_MAP.get(lang_code)}")
            except Exception as lang_err:
                # If we can't get language info, just proceed without it
                if st.session_state.get('debug_mode', False):
                    st.sidebar.warning(f"Language detection issue: {lang_err}")
                debug_log(f"Debug - Language detection failed with error: {lang_err}")
                traceback.print_exc()  # Print the full error traceback
                
                # Last attempt to extract language from string representation
                try:
                    info_str = str(info)
                    if "language='" in info_str:
                        # Try to extract language from string representation
                        lang_start = info_str.find("language='") + 10
                        lang_end = info_str.find("'", lang_start)
                        if lang_start > 10 and lang_end > lang_start:
                            extracted_lang = info_str[lang_start:lang_end]
                            debug_log(f"Debug - Extracted language from string: '{extracted_lang}'")
                            detected_language = extracted_lang.title()

                except Exception as e:
                    debug_log(f"Debug - Failed to extract language from string: {e}")
                
                # Is this setting language to None?
                detected_language = None
            
            # Combine all segments into one text
            # For newer versions, segments might be an iterable object
            try:
                transcription = " ".join([s.text for s in segments])
            except TypeError:
                # If segments is not iterable, it might be a single result or different format
                if hasattr(segments, 'text'):
                    # Single segment result
                    transcription = segments.text
                else:
                    # Last resort, try to convert to string
                    transcription = str(segments)
                    
        except Exception as e:
            st.warning(f"Whisper transcription failed for segment: {e}")
            transcription = ""
            detected_language = None
        finally:
            # Safely remove the temporary file
            safe_remove_file(temp_filepath)
            
        return transcription.strip(), detected_language
    except Exception as e:
        st.error(f"Error during Whisper segment transcription: {e}")
        traceback.print_exc()
        return "", None

def transcribe_wav2vec2_segment(audio_array, start_time, end_time, model, processor, sample_rate=16000):
    """Transcribe a specific audio segment using Wav2Vec2"""
    try:
        # Calculate the indices in the audio array
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        
        # Safety check for bounds
        if start_idx >= len(audio_array) or end_idx > len(audio_array):
            st.warning(f"Segment indices out of bounds: {start_idx}-{end_idx}, array length: {len(audio_array)}")
            return "", None
        
        # Extract the segment
        segment = audio_array[start_idx:end_idx]
        
        # If segment is too short, return empty string
        if len(segment) < 0.3 * sample_rate:  # Less than 300ms
            return "", None
        
        # Process with ASR - explicitly create attention mask
        inputs = processor(segment, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        
        # Always create a proper attention mask
        attention_mask = torch.ones(inputs.input_values.shape[:2] if len(inputs.input_values.shape) > 2 else (1, inputs.input_values.shape[1]), dtype=torch.long)
        
        # Move inputs to GPU if model is on GPU
        device = next(model.parameters()).device
        input_values = inputs.input_values.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            try:
                logits = model(input_values, attention_mask=attention_mask).logits
                
                # Get predicted ids and convert to text
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
                
                return transcription, None
            except Exception as e:
                st.warning(f"Wav2Vec2 inference failed: {e}")
                return "", None
                
    except Exception as e:
        st.error(f"Error during Wav2Vec2 segment transcription: {e}")
        traceback.print_exc()
        return "", None

# Main upload area with additional instructions
st.title("Meeting Transcription with Speaker Diarization and Translation")
st.subheader("Upload your meeting recording")

# Settings in sidebar
with st.sidebar:
    # Performance Meter - add at the top of the sidebar
    meter_header_col1, meter_header_col2 = st.columns([3, 1])
    with meter_header_col1:
        st.markdown("## Performance Meter")
    with meter_header_col2:
        if st.button("🔄", help="Refresh metrics"):
            st.session_state.memory_needs_update = True
            # Remove this rerun call as we'll rely on natural UI updates
    
    # Update the performance metrics on initial load and when needed
    current_time = time.time()
    
    # Update metrics every 10 seconds or when explicitly triggered
    should_update = (current_time - st.session_state.last_memory_update > 10) or st.session_state.memory_needs_update
    
    # GPU/CPU selection - define this early because we need it for the performance meter
    use_gpu = False
    gpu_type = None
    
    # Check available GPU types and provide appropriate UI
    if cuda_available or mps_available or rocm_available:
        use_gpu = st.checkbox(
            "Use GPU acceleration", 
            value=True,
            help="Run models on GPU for faster processing (requires compatible GPU)"
        )
        
        # Determine GPU type based on availability
        if use_gpu:
            if cuda_available:
                gpu_type = "nvidia"
            elif mps_available:
                gpu_type = "apple"
            elif rocm_available:
                gpu_type = "amd"
    
    # Get memory metrics based on GPU type
    metrics = get_memory_metrics(use_gpu, gpu_type)
    
    # Use containers instead of direct calls to allow updating
    metrics_container = st.container()
    with metrics_container:
        # Store reference to this container in session state for updates
        if 'metrics_placeholder' not in st.session_state:
            st.session_state.metrics_placeholder = metrics_container
        
        # Create a progress bar for memory usage
        if use_gpu:
            if gpu_type == "nvidia" and cuda_available:
                st.markdown(f"**GPU Memory** ({torch.cuda.get_device_name(0)})")
                mem_bar = st.progress(float(metrics["usage_percent"]) / 100)
                st.caption(f"{metrics['used']:.1f}GB used / {metrics['total']:.1f}GB total ({metrics['usage_percent']:.1f}%)")
            elif gpu_type == "apple" and mps_available:
                model_name = metrics.get("model", "Apple Silicon")
                st.markdown(f"**GPU Memory** ({model_name})")
                mem_bar = st.progress(float(metrics["usage_percent"]) / 100)
                st.caption(f"{metrics['used']:.1f}GB used / {metrics['total']:.1f}GB total ({metrics['usage_percent']:.1f}%)")
                st.caption("Note: Memory values are estimates for Apple Silicon")
            elif gpu_type == "amd" and rocm_available:
                model_name = metrics.get("model", "AMD GPU")
                st.markdown(f"**GPU Memory** ({model_name})")
                mem_bar = st.progress(float(metrics["usage_percent"]) / 100)
                st.caption(f"{metrics['used']:.1f}GB used / {metrics['total']:.1f}GB total ({metrics['usage_percent']:.1f}%)")
            
            # Show data source and additional details if debug mode is on
            if st.session_state.debug_mode:
                st.caption(f"Source: {metrics['source']}")
                if 'allocated' in metrics:
                    st.caption(f"Allocated: {metrics['allocated']:.1f}GB | Used: {metrics['used']:.1f}GB")
        else:
            st.markdown("**System RAM**")
            mem_bar = st.progress(float(metrics["usage_percent"]) / 100)
            st.caption(f"{metrics['used']:.1f}GB used / {metrics['total']:.1f}GB total ({metrics['usage_percent']:.1f}%)")
    
    # Store the last update time and reset update flag
    st.session_state.last_memory_update = current_time
    st.session_state.memory_needs_update = False
    
    # Settings header
    st.header("Settings")
    
    # Transcription engine selection - always include Seamless
    transcription_engine = st.radio(
        "Transcription Engine:",
        ["Whisper (Optimized)", "Wav2Vec2", "Seamless M4T"],
        index=0,
        help="Select the speech recognition model"
    )
    
    # Preferred language selection - use dynamic list based on engine
    if transcription_engine == "Seamless M4T":
        # Seamless supports many languages for text-to-text translation
        seamless_translation_languages = [
            "English", "Spanish", "French", "German", "Italian", "Chinese", "Chinese (Traditional)", 
            "Arabic", "Hindi", "Russian", "Turkish", "Portuguese", "Dutch", "Japanese", "Korean", 
            "Polish", "Romanian", "Ukrainian", "Vietnamese", "Thai", "Indonesian", "Malay", 
            "Bengali", "Urdu", "Czech", "Hungarian", "Finnish", "Swedish", "Norwegian", "Danish", 
            "Greek", "Bulgarian", "Croatian", "Serbian", "Slovak", "Slovenian", "Estonian", 
            "Latvian", "Lithuanian", "Albanian", "Armenian", "Azerbaijani", "Georgian", 
            "Kazakh", "Macedonian", "Mongolian", "Persian", "Swahili", "Tamil", "Telugu",
            "Afrikaans", "Amharic", "Catalan", "Welsh", "Gujarati", "Hebrew", "Icelandic",
            "Irish", "Kannada", "Khmer", "Lao", "Malayalam", "Marathi", "Nepali", "Punjabi", 
            "Tagalog", "Yoruba", "Zulu", "Maltese"
        ]
        preferred_languages = sorted(seamless_translation_languages)
    elif transcription_engine == "Whisper (Optimized)":
        # For Whisper, we use Neural MT which supports a smaller set of languages
        nmt_translation_languages = [
            "English", "Spanish", "French", "German", "Italian", "Portuguese", 
            "Chinese", "Japanese", "Korean", "Arabic", "Russian", "Turkish",
            "Dutch", "Polish", "Romanian", "Ukrainian", "Czech", "Hungarian",
            "Finnish", "Swedish", "Greek", "Bulgarian", "Danish", "Norwegian",
            "Catalan", "Hebrew", "Indonesian", "Thai", "Vietnamese", "Hindi",
            "Bengali", "Tamil", "Telugu", "Urdu", "Swahili", "Welsh"
        ]
        preferred_languages = sorted(nmt_translation_languages)
    else:  # Wav2Vec2
        # For Wav2Vec2, just English as default
        preferred_languages = ["English"]
    
    # Update preferred language, defaulting to English if current selection isn't available
    if st.session_state.preferred_language not in preferred_languages:
        st.session_state.preferred_language = "English"
    
    st.session_state.preferred_language = st.selectbox(
        "Preferred Language:", 
        preferred_languages,
        index=preferred_languages.index(st.session_state.preferred_language),
        help="Content in other languages will be offered for translation to this language"
    )
    
    # GPU/CPU selection info display
    if use_gpu:
        if gpu_type == "nvidia":
            model_name = f"{torch.cuda.get_device_name(0)}"
            st.success(f"✓ GPU: {model_name}")
        elif gpu_type == "apple":
            model_name = metrics.get("model", "Apple Silicon")
            st.success(f"✓ GPU: {model_name}")
        elif gpu_type == "amd":
            model_name = metrics.get("model", "AMD GPU")
            st.success(f"✓ GPU: {model_name}")
        else:
            st.info("GPU type not detected. Running on CPU.")
    else:
        if cuda_available or mps_available or rocm_available:
            st.info("Running on CPU (enable GPU for faster processing)")
        else:
            st.info("No compatible GPU detected. Running on CPU.")
    
    # Model selection based on engine
    if transcription_engine == "Whisper (Optimized)":
        whisper_model_options = {
            "tiny": "Tiny (Fast, less accurate)",
            "base": "Base (Good balance)",
            "small": "Small (Better quality)",
            "medium": "Medium (High quality)",
            "large-v3": "Large v3 (Best quality, slower)"
        }
        
        selected_model = st.selectbox(
            "Select Whisper model size:",
            list(whisper_model_options.keys()),
            index=2,
            format_func=lambda x: whisper_model_options[x],
            help="All models run locally for security. Larger models are more accurate but slower."
        )
        
        # Language selection - update to use the comprehensive list
        languages = ["Auto-detect"] + sorted(list(WHISPER_LANGUAGE_MAP.values()))
        selected_language = st.selectbox(
            "Language (optional):", 
            languages, 
            index=0,
            help="Selecting a specific language can improve transcription accuracy"
        )
        
    elif transcription_engine == "Wav2Vec2":
        wav2vec2_model_options = {
            "facebook/wav2vec2-base-960h": "English (Base)",
            "facebook/wav2vec2-large-960h-lv60-self": "English (Large)",
            "facebook/wav2vec2-large-robust-ft-swbd-300h": "English (Robust)",
            "facebook/wav2vec2-xls-r-300m": "XLS-R 300M (Multilingual)",
            "facebook/wav2vec2-large-xlsr-53": "XLS-R 53 (Multilingual)",
            "facebook/mms-1b-all": "MMS 1B (Multilingual)"
        }
        
        selected_model = st.selectbox(
            "Select Wav2Vec2 model:",
            list(wav2vec2_model_options.keys()),
            index=0,
            format_func=lambda x: wav2vec2_model_options[x],
            help="All models run locally. Wav2Vec2 outputs are uppercase without punctuation."
        )
    
    elif transcription_engine == "Seamless M4T":
        # Detect OS platform
        system_platform = platform.system().lower()
        
        try:
            # Different initialization based on platform
            if system_platform == "linux":
                # Linux - Use seamless_communication package if available
                try:
                    from seamless_communication.inference import Translator
                    
                    # Seamless M4T specific options
                    seamless_model_options = {
                        "facebook/seamless-m4t-v2-large": "Seamless M4T v2 Large"
                    }
                    
                    selected_model = st.selectbox(
                        "Select Seamless M4T model:",
                        list(seamless_model_options.keys()),
                        index=0,
                        format_func=lambda x: seamless_model_options[x],
                        help="Seamless M4T models support speech recognition and translation."
                    )
                    
                    # Language selection
                    seamless_languages = [
                        "English", "Spanish", "French", "German", "Italian", "Chinese", "Chinese (Traditional)", 
                        "Arabic", "Hindi", "Russian", "Turkish", "Portuguese", "Dutch", "Japanese", "Korean", 
                        "Polish", "Romanian", "Ukrainian", "Vietnamese", "Thai", "Indonesian", "Malay", 
                        "Bengali", "Urdu", "Czech", "Hungarian", "Finnish", "Swedish", "Norwegian", "Danish", 
                        "Greek", "Bulgarian", "Croatian", "Serbian", "Slovak", "Slovenian", "Estonian", 
                        "Latvian", "Lithuanian", "Albanian", "Armenian", "Azerbaijani", "Georgian", 
                        "Kazakh", "Macedonian", "Mongolian", "Persian", "Swahili", "Tamil", "Telugu",
                        "Afrikaans", "Amharic", "Catalan", "Welsh", "Gujarati", "Hebrew", "Icelandic",
                        "Irish", "Kannada", "Khmer", "Lao", "Malayalam", "Marathi", "Nepali", "Punjabi", 
                        "Sinhala", "Tagalog", "Yoruba", "Zulu", "Maltese"
                    ]
                    transcription_language = st.selectbox(
                        "Transcribe to language:", 
                        sorted(seamless_languages), 
                        index=sorted(seamless_languages).index("English"),
                        help="Language to transcribe audio into"
                    )
                    
                    # Source language with auto-detect option
                    source_languages = ["Auto-detect"] + sorted(seamless_languages)
                    source_language = st.selectbox(
                        "Source language (if known):",
                        source_languages,
                        index=0,
                        help="Language spoken in the audio (Auto-detect works well for most cases)"
                    )
                except ImportError:
                    st.error("""
                    ### The seamless_communication package is not installed properly on Linux
                    
                    Please run the installation script again:
                    ```
                    ./run.sh
                    ```
                    """)
                    # Provide a dummy selectbox to prevent UI errors
                    st.selectbox("Select model:", ["Package not available"], disabled=True)
            else:
                # Windows/macOS - Use transformers implementation
                try:
                    from transformers import AutoProcessor, SeamlessM4Tv2Model
                    
                    # Seamless M4T v2 options
                    selected_model = "facebook/seamless-m4t-v2-large"
                    st.info(f"Using Hugging Face Transformers with model: {selected_model}")
                    
                    # Add debug mode for transformers
                    if 'transformers_debug' not in st.session_state:
                        st.session_state.transformers_debug = False
                    
                    st.session_state.transformers_debug = st.checkbox(
                        "Debug Transformers model issues", 
                        value=st.session_state.transformers_debug,
                        help="Enable detailed output for troubleshooting model problems"
                    )
                    
                    # Language selection
                    seamless_languages = [
                        "English", "Spanish", "French", "German", "Italian", "Chinese", "Chinese (Traditional)", 
                        "Arabic", "Hindi", "Russian", "Turkish", "Portuguese", "Dutch", "Japanese", "Korean", 
                        "Polish", "Romanian", "Ukrainian", "Vietnamese", "Thai", "Indonesian", "Malay", 
                        "Bengali", "Urdu", "Czech", "Hungarian", "Finnish", "Swedish", "Norwegian", "Danish", 
                        "Greek", "Bulgarian", "Croatian", "Serbian", "Slovak", "Slovenian", "Estonian", 
                        "Latvian", "Lithuanian", "Albanian", "Armenian", "Azerbaijani", "Georgian", 
                        "Kazakh", "Macedonian", "Mongolian", "Persian", "Swahili", "Tamil", "Telugu",
                        "Afrikaans", "Amharic", "Catalan", "Welsh", "Gujarati", "Hebrew", "Icelandic",
                        "Irish", "Kannada", "Khmer", "Lao", "Malayalam", "Marathi", "Nepali", "Punjabi", 
                        "Sinhala", "Tagalog", "Yoruba", "Zulu", "Maltese"
                    ]
                    transcription_language = st.selectbox(
                        "Transcribe to language:", 
                        sorted(seamless_languages), 
                        index=sorted(seamless_languages).index("English"),
                        help="Language to transcribe audio into"
                    )
                    
                    # Source language with auto-detect option
                    source_languages = ["Auto-detect"] + sorted(seamless_languages)
                    source_language = st.selectbox(
                        "Source language (if known):",
                        source_languages,
                        index=0,
                        help="Language spoken in the audio (Auto-detect works well for most cases)"
                    )
                except ImportError:
                    st.error("""
                    ### The transformers library is not installed properly
                    
                    Please run the installation script:
                    ```
                    run.bat  # On Windows
                    ./run.sh  # On macOS
                    ```
                    """)
                    # Provide a dummy selectbox to prevent UI errors
                    st.selectbox("Select model:", ["Package not available"], disabled=True)
                    
        except Exception as e:
            st.error(f"Error initializing Seamless M4T interface: {e}")
            # Provide a dummy selectbox to prevent UI errors
            st.selectbox("Select model:", ["Error loading interface"], disabled=True)
    
    # Advanced options
    with st.expander("Advanced Options"):
        min_segment_duration = st.slider(
            "Minimum segment duration (seconds)", 
            min_value=0.5, 
            max_value=5.0, 
            value=1.5,
            help="Shorter segments will be skipped"
        )
        
        use_librosa = st.checkbox(
            "Use librosa for audio loading", 
            value=True, 
            help="May be more reliable but slightly slower"
        )
        
        clean_temp_files = st.checkbox(
            "Clean temporary files", 
            value=True,
            help="Attempt to remove temporary files after processing"
        )
        
        # Display debug information
        st.session_state.debug_mode = st.checkbox(
            "Show debug information", 
            value=st.session_state.debug_mode,
            help="Display processing details in the sidebar"
        )

# Main upload area with additional instructions
st.info("Upload an audio recording of a meeting or conversation. The system will identify different speakers and transcribe what each person says.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac", "ogg"])

if uploaded_file is not None:
    # Calculate a hash of the uploaded file to check if it's a new file
    file_content = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_content).hexdigest()
    
    # Check if it's a new file
    is_new_file = st.session_state.current_file_hash != file_hash
    
    if is_new_file:
        # Clear any existing audio samples since they belong to the previous file
        st.session_state.audio_samples = {}
        st.session_state.current_file_hash = file_hash
        st.session_state.transcript_data = None
    
    # Only process the audio if this is a new file or we don't have transcript data yet
    if is_new_file or st.session_state.transcript_data is None:
        with st.spinner("Processing audio..."):
            # Create a unique temp file name to avoid conflicts
            temp_filename = f"input_audio_{uuid.uuid4().hex}{os.path.splitext(uploaded_file.name)[1]}"
            temp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
            
            # Save uploaded file to the temporary location
            with open(temp_file_path, 'wb') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
            
            # Load the audio for processing
            try:
                # Load models based on selection
                with st.spinner("Loading models..."):
                    diarization_model = load_diarization_model(use_gpu)
                    
                    if transcription_engine == "Whisper (Optimized)":
                        transcription_model = load_whisper_model(selected_model, use_gpu)
                        model_type = "whisper"
                    elif transcription_engine == "Wav2Vec2":
                        transcription_model, processor = load_wav2vec2_model(selected_model, use_gpu)
                        model_type = "wav2vec2"
                    elif transcription_engine == "Seamless M4T":
                        # Use the appropriate implementation based on platform
                        system_platform = platform.system().lower()
                        
                        if system_platform == "linux":
                            # Linux: Use seamless_communication
                            try:
                                from seamless_communication.inference import Translator
                                
                                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                                torch_dtype = torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32
                                
                                # Loading feedback
                                st.info(f"Loading Seamless M4T model on {device} using seamless_communication...")
                                
                                try:
                                    # Load the translator with the selected model
                                    translator = Translator(
                                        selected_model,
                                        "vocoder_36langs",
                                        device,
                                        torch_dtype
                                    )
                                    
                                    # Store as model for transcription
                                    transcription_model = translator
                                    processor = None  # Not needed for seamless_communication
                                    model_type = "seamless"
                                    
                                    # Get language codes for later
                                    language_map = {
                                        "English": "eng", "Spanish": "spa", "French": "fra", "German": "deu", 
                                        "Italian": "ita", "Chinese": "cmn", "Chinese (Traditional)": "cmn_Hant",
                                        "Arabic": "arb", "Hindi": "hin", "Russian": "rus", "Turkish": "tur",
                                        "Portuguese": "por", "Dutch": "nld", "Japanese": "jpn", "Korean": "kor",
                                        "Polish": "pol", "Romanian": "ron", "Ukrainian": "ukr", "Vietnamese": "vie",
                                        "Thai": "tha", "Indonesian": "ind", "Malay": "zlm", "Bengali": "ben",
                                        "Urdu": "urd", "Czech": "ces", "Hungarian": "hun", "Finnish": "fin",
                                        "Swedish": "swe", "Norwegian": "nob", "Danish": "dan", "Greek": "ell",
                                        "Bulgarian": "bul", "Croatian": "hrv", "Serbian": "srp", "Slovak": "slk",
                                        "Slovenian": "slv", "Estonian": "est", "Latvian": "lvs", "Lithuanian": "lit",
                                        "Albanian": "sqi", "Armenian": "hye", "Azerbaijani": "azj", "Georgian": "kat",
                                        "Kazakh": "kaz", "Macedonian": "mkd", "Mongolian": "khk", "Persian": "pes",
                                        "Swahili": "swh", "Tamil": "tam", "Telugu": "tel", "Afrikaans": "afr",
                                        "Amharic": "amh", "Catalan": "cat", "Welsh": "cym", "Gujarati": "guj",
                                        "Hebrew": "heb", "Icelandic": "isl", "Irish": "gle", "Kannada": "kan",
                                        "Khmer": "khm", "Lao": "lao", "Malayalam": "mal", "Marathi": "mar",
                                        "Nepali": "npi", "Punjabi": "pan", "Sinhala": "sin", "Tagalog": "tgl",
                                        "Yoruba": "yor", "Zulu": "zul", "Maltese": "mlt"
                                    }
                                    selected_language_code = language_map.get(transcription_language, "eng")
                                    
                                    # Handle source language
                                    if source_language != "Auto-detect":
                                        selected_source_code = language_map.get(source_language, None)
                                    else:
                                        selected_source_code = None  # Auto-detect
                                        
                                except Exception as e:
                                    st.error(f"Failed to load Seamless M4T: {e}")
                                    st.warning("Falling back to Whisper model.")
                                    
                                    # Fall back to Whisper
                                    transcription_model = load_whisper_model("small", use_gpu)
                                    model_type = "whisper"
                                    language = None  # Default to auto-detect
                            except ImportError as e:
                                st.error(f"Failed to import seamless_communication on Linux: {e}.")
                                st.warning("Falling back to Whisper model.")
                                
                                # Fall back to Whisper
                                transcription_model = load_whisper_model("small", use_gpu)
                                model_type = "whisper"
                                language = None  # Default to auto-detect
                        else:
                            # Windows/macOS: Use transformers implementation
                            try:
                                from transformers import AutoProcessor, SeamlessM4Tv2Model
                                
                                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                                
                                # Loading feedback
                                st.info(f"Loading Seamless M4T v2 model on {device} using transformers...")
                                
                                try:
                                    # Load processor and model from Hugging Face
                                    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
                                    
                                    # Show debug info if enabled
                                    if st.session_state.transformers_debug:
                                        st.write("Processor type:", type(processor))
                                        st.write("Available processor attributes:", dir(processor))
                                    
                                    # Set return_dict=True to get proper outputs
                                    model = SeamlessM4Tv2Model.from_pretrained(
                                        "facebook/seamless-m4t-v2-large", 
                                        torch_dtype=torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32,
                                        return_dict=True
                                    )
                                    
                                    # Move to appropriate device
                                    if use_gpu and torch.cuda.is_available():
                                        model = model.to("cuda")
                                    
                                    # Create a wrapper tuple to match the expected format
                                    transcription_model = (model, processor)
                                    model_type = "transformers_seamless"
                                    
                                    # Get language codes for later
                                    language_map = {
                                        "English": "eng", "Spanish": "spa", "French": "fra", "German": "deu", 
                                        "Italian": "ita", "Chinese": "cmn", "Chinese (Traditional)": "cmn_Hant",
                                        "Arabic": "arb", "Hindi": "hin", "Russian": "rus", "Turkish": "tur",
                                        "Portuguese": "por", "Dutch": "nld", "Japanese": "jpn", "Korean": "kor",
                                        "Polish": "pol", "Romanian": "ron", "Ukrainian": "ukr", "Vietnamese": "vie",
                                        "Thai": "tha", "Indonesian": "ind", "Malay": "zlm", "Bengali": "ben",
                                        "Urdu": "urd", "Czech": "ces", "Hungarian": "hun", "Finnish": "fin",
                                        "Swedish": "swe", "Norwegian": "nob", "Danish": "dan", "Greek": "ell",
                                        "Bulgarian": "bul", "Croatian": "hrv", "Serbian": "srp", "Slovak": "slk",
                                        "Slovenian": "slv", "Estonian": "est", "Latvian": "lvs", "Lithuanian": "lit",
                                        "Albanian": "sqi", "Armenian": "hye", "Azerbaijani": "azj", "Georgian": "kat",
                                        "Kazakh": "kaz", "Macedonian": "mkd", "Mongolian": "khk", "Persian": "pes",
                                        "Swahili": "swh", "Tamil": "tam", "Telugu": "tel", "Afrikaans": "afr",
                                        "Amharic": "amh", "Catalan": "cat", "Welsh": "cym", "Gujarati": "guj",
                                        "Hebrew": "heb", "Icelandic": "isl", "Irish": "gle", "Kannada": "kan",
                                        "Khmer": "khm", "Lao": "lao", "Malayalam": "mal", "Marathi": "mar",
                                        "Nepali": "npi", "Punjabi": "pan", "Sinhala": "sin", "Tagalog": "tgl",
                                        "Yoruba": "yor", "Zulu": "zul", "Maltese": "mlt"
                                    }
                                    selected_language_code = language_map.get(transcription_language, "eng")
                                    
                                    # Handle source language
                                    if source_language != "Auto-detect":
                                        selected_source_code = language_map.get(source_language, None)
                                    else:
                                        selected_source_code = None  # Auto-detect
                                    
                                except Exception as e:
                                    st.error(f"Failed to load Seamless M4T with transformers: {e}")
                                    st.warning("Falling back to Whisper model.")
                                    
                                    # Fall back to Whisper
                                    transcription_model = load_whisper_model("small", use_gpu)
                                    model_type = "whisper"
                                    language = None  # Default to auto-detect
                            except ImportError as e:
                                st.error(f"Failed to import transformers on {system_platform}: {e}.")
                                st.warning("Falling back to Whisper model.")
                                
                                # Fall back to Whisper
                                transcription_model = load_whisper_model("small", use_gpu)
                                model_type = "whisper"
                                language = None  # Default to auto-detect
                    else:
                        # This branch should never be reached now that we've removed "Other Models"
                        st.error("Unknown transcription engine selected")
                        transcription_model = load_whisper_model("small", use_gpu)
                        model_type = "whisper"
                        language = None
                    
                    if st.session_state.debug_mode:
                        st.sidebar.success(f"✓ Model loaded successfully ({transcription_engine})")
                
                # Load audio data
                try:
                    if use_librosa:
                        # Use librosa for more reliable loading
                        audio_array, sample_rate = librosa.load(temp_file_path, sr=16000)
                    else:
                        # Use torchaudio with explicit backend parameter
                        try:
                            # Always use explicit backend to avoid warning
                            waveform, sample_rate = torchaudio.load(temp_file_path, backend="soundfile")
                        except Exception:
                            # Fallback with default backend, but still proper parameters
                            waveform, sample_rate = torchaudio.load(temp_file_path)
                            
                        if waveform.shape[0] > 1:  # Convert to mono if stereo
                            waveform = torch.mean(waveform, dim=0, keepdim=True)
                        
                        # Resample if needed using newer method
                        if sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(
                                orig_freq=sample_rate, 
                                new_freq=16000
                            )
                            waveform = resampler(waveform)
                            sample_rate = 16000
                        
                        # Convert to numpy for librosa compatibility
                        audio_array = waveform.squeeze().numpy()
                    
                    if st.session_state.debug_mode:
                        st.sidebar.success(f"✓ Audio loaded successfully: {len(audio_array)/16000:.2f}s at {sample_rate}Hz")
                
                    # Store audio data in session state for later use in translation
                    st.session_state.audio_data = {
                        "array": audio_array,
                        "sample_rate": sample_rate,
                        "file_path": temp_file_path
                    }
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
                    st.info("Trying fallback audio loading method...")
                    audio_array, sample_rate = librosa.load(temp_file_path, sr=16000)
                
                # Create PyAnnote audio object for diarization
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Use PyAnnote Audio with explicit parameters to avoid warnings
                    audio = paa.Audio(mono='downmix', sample_rate=16000)
                    waveform, sr = audio(temp_file_path)
                    file_dict = {"waveform": waveform, "sample_rate": sr}
                
                # Perform diarization
                with st.spinner("Identifying speakers..."):
                    diarization_results = diarization_model(file_dict)
                    
                    # Get total speakers identified
                    speakers = set([speaker for _, _, speaker in diarization_results.itertracks(yield_label=True)])
                    
                    if st.session_state.debug_mode:
                        st.sidebar.success(f"✓ Detected {len(speakers)} speakers")
                        # Display a sample of the diarization results
                        if len(speakers) > 0:
                            st.sidebar.text("Speaker segments (sample):")
                            count = 0
                            for segment, _, speaker in diarization_results.itertracks(yield_label=True):
                                if count < 5:  # Show just 5 segments
                                    st.sidebar.text(f"[{segment.start:.2f}-{segment.end:.2f}] {speaker}")
                                    count += 1
                                else:
                                    break
                
                # For segment-specific detected languages, store a dictionary to track them
                segment_languages = {}

                # Get selected language code for Whisper
                language = None
                if transcription_engine == "Whisper (Optimized)" and selected_language != "Auto-detect":
                    language = selected_language

                # Initialize the transcript list
                full_transcript = []

                # Process each diarization segment
                segment_count = 0
                total_segments = sum(1 for _ in diarization_results.itertracks(yield_label=True))
                error_count = 0  # Initialize error count
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for segment, track, speaker in diarization_results.itertracks(yield_label=True):
                    start_time = segment.start
                    end_time = segment.end
                    
                    # Ensure we have valid segment boundaries
                    if end_time <= start_time:
                        if st.session_state.debug_mode:
                            st.sidebar.warning(f"Invalid segment times: {start_time:.2f}-{end_time:.2f}")
                        segment_count += 1
                        progress_bar.progress(segment_count / total_segments)
                        continue
                    
                    # Skip very short segments
                    if end_time - start_time < min_segment_duration:
                        if st.session_state.debug_mode:
                            st.sidebar.text(f"Skipped short segment: {start_time:.2f}-{end_time:.2f}")
                        segment_count += 1
                        progress_bar.progress(segment_count / total_segments)
                        continue
                    
                    # Update progress
                    segment_count += 1
                    progress_percentage = segment_count / total_segments
                    progress_bar.progress(progress_percentage)
                    status_text.text(f"Transcribing segment {segment_count}/{total_segments}: [{start_time:.2f}-{end_time:.2f}] Speaker {speaker}")
                    
                    try:
                        # Transcribe segment based on selected engine
                        if model_type == "whisper":
                            segment_text, detected_language = transcribe_whisper_segment(
                                temp_file_path, 
                                start_time, 
                                end_time, 
                                transcription_model,
                                language=language
                            )
                            
                            # Debug print to verify detected language
                            debug_log(f"Segment {segment_count} - Detected Language from Whisper: {detected_language}")
                            
                        elif model_type == "wav2vec2":
                            segment_text, detected_language = transcribe_wav2vec2_segment(
                                audio_array,
                                start_time,
                                end_time,
                                transcription_model,
                                processor
                            )
                        elif model_type == "seamless":
                            # Create a temporary file for this segment
                            temp_segment = audio_array[int(start_time * sample_rate):int(end_time * sample_rate)]
                            
                            # Only process if segment has enough content
                            if len(temp_segment) > 0.5 * sample_rate:
                                # Create a temporary WAV file for the segment
                                segment_filename = f"seamless_segment_{uuid.uuid4().hex}.wav"
                                segment_filepath = os.path.join(tempfile.gettempdir(), segment_filename)
                                
                                try:
                                    # Write the audio segment to a file
                                    sf.write(segment_filepath, temp_segment, sample_rate)
                                    
                                    # First check if the target language is directly supported for speech-to-text
                                    # We'll try to do speech-to-text directly if supported
                                    try:
                                        # Prepare arguments for Seamless M4T translator
                                        predict_args = {
                                            "input": segment_filepath,
                                            "task_str": "S2TT",  # Speech to text transcription
                                            "tgt_lang": selected_language_code,  # Use selected language
                                        }
                                        
                                        # Add source language if specified
                                        if selected_source_code is not None:
                                            predict_args["src_lang"] = selected_source_code
                                        
                                        # Use Seamless M4T translator in ASR mode
                                        result = transcription_model.predict(**predict_args)
                                        segment_text = result.text
                                        
                                    except Exception as e:
                                        # If we get an error about unsupported target language, use a two-step approach
                                        if "is not supported by this model" in str(e):
                                            # Step 1: Transcribe to English first (or another well-supported language)
                                            transcribe_args = {
                                                "input": segment_filepath,
                                                "task_str": "S2TT",  # Speech to text transcription
                                                "tgt_lang": "eng",  # Transcribe to English
                                            }
                                            
                                            # Add source language if specified
                                            if selected_source_code is not None:
                                                transcribe_args["src_lang"] = selected_source_code
                                                
                                            # Transcribe to English
                                            eng_result = transcription_model.predict(**transcribe_args)
                                            eng_text = eng_result.text
                                            
                                            # Step 2: Translate English text to target language
                                            translate_args = {
                                                "input": eng_text,
                                                "task_str": "T2TT",  # Text to text translation
                                                "tgt_lang": selected_language_code,  # Target language
                                                "src_lang": "eng",  # Source is English
                                            }
                                            
                                            # Translate from English to target language
                                            translate_result = transcription_model.predict(**translate_args)
                                            segment_text = translate_result.text
                                            
                                            if st.session_state.debug_mode:
                                                st.sidebar.info(f"Used two-step transcription for {selected_language_code}: Speech→English→{transcription_language}")
                                        else:
                                            # If it's some other error, re-raise it
                                            raise
                                    
                                    # For source language, use the detected language if available
                                    detected_language = None
                                    if hasattr(result, 'src_lang'):
                                        # Map language code to full name
                                        language_map = {
                                            "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German", 
                                            "ita": "Italian", "cmn": "Chinese", "arb": "Arabic", "hin": "Hindi",
                                            "rus": "Russian", "tur": "Turkish"
                                        }
                                        detected_language = language_map.get(result.src_lang, result.src_lang.title())
                                    elif selected_source_code:
                                        # If we specified a source language, use that
                                        language_map = {
                                            "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German", 
                                            "ita": "Italian", "cmn": "Chinese", "arb": "Arabic", "hin": "Hindi",
                                            "rus": "Russian", "tur": "Turkish"
                                        }
                                        detected_language = language_map.get(selected_source_code, source_language)
                                    else:
                                        # Fallback to the source language from UI selection
                                        detected_language = source_language if source_language != "Auto-detect" else "English"
                                        
                                finally:
                                    # Clean up the temporary file
                                    safe_remove_file(segment_filepath)
                            else:
                                segment_text = ""
                                detected_language = None
                        elif model_type == "transformers_seamless":
                            # Use transformers-based Seamless M4T v2 model
                            model, processor = transcription_model
                            
                            # Extract the audio segment
                            temp_segment = audio_array[int(start_time * sample_rate):int(end_time * sample_rate)]
                            
                            # Only process if segment has enough content
                            if len(temp_segment) > 0.5 * sample_rate:
                                try:
                                    # Create input for the model
                                    inputs = processor(
                                        audios=torch.tensor(temp_segment).unsqueeze(0), 
                                        sampling_rate=sample_rate, 
                                        return_tensors="pt"
                                    )
                                    
                                    # Debug info if enabled
                                    if st.session_state.transformers_debug:
                                        st.sidebar.markdown(f"**Debug: Segment {segment_count}**")
                                        st.sidebar.write(f"Input shape: {torch.tensor(temp_segment).shape}")
                                        st.sidebar.write(f"Input keys: {list(inputs.keys())}")
                                    
                                    # Move to appropriate device
                                    device = next(model.parameters()).device
                                    for key in inputs:
                                        if torch.is_tensor(inputs[key]):
                                            inputs[key] = inputs[key].to(device)
                                    
                                    # Run generation with target language
                                    with torch.no_grad():
                                        try:
                                            # Try direct speech-to-text generation with the target language
                                            try:
                                                # Generate text from speech
                                                # For text output we use model.generate with generate_speech=False
                                                output_tokens = model.generate(
                                                    **inputs,
                                                    tgt_lang=selected_language_code,
                                                    generate_speech=False,  # Important: only generate text
                                                    return_intermediate_token_ids=False,
                                                    return_dict_in_generate=True  # Ensure we get a structured output
                                                )
                                            except Exception as lang_error:
                                                # Check if the error is due to unsupported language
                                                if "is not supported by this model" in str(lang_error):
                                                    if st.session_state.transformers_debug:
                                                        st.sidebar.warning(f"Target language not directly supported: {lang_error}")
                                                    
                                                    # Step 1: First generate transcription in English
                                                    eng_output = model.generate(
                                                        **inputs,
                                                        tgt_lang="eng",  # Transcribe to English first
                                                        generate_speech=False,
                                                        return_intermediate_token_ids=False,
                                                        return_dict_in_generate=True
                                                    )
                                                    
                                                    # Decode the English output
                                                    if hasattr(eng_output, 'sequences'):
                                                        eng_text = processor.batch_decode(eng_output.sequences, skip_special_tokens=True)[0]
                                                    else:
                                                        eng_text = processor.decode(eng_output[0], skip_special_tokens=True)
                                                    
                                                    # Step 2: Translate English text to target language
                                                    # Create text input for translation
                                                    text_inputs = processor(
                                                        text=eng_text, 
                                                        src_lang="eng", 
                                                        return_tensors="pt"
                                                    )
                                                    
                                                    # Move to device
                                                    for key in text_inputs:
                                                        if torch.is_tensor(text_inputs[key]):
                                                            text_inputs[key] = text_inputs[key].to(device)
                                                    
                                                    # Translate to target language
                                                    translate_output = model.generate(
                                                        **text_inputs,
                                                        tgt_lang=selected_language_code,
                                                        generate_speech=False,
                                                        return_intermediate_token_ids=False,
                                                        return_dict_in_generate=True
                                                    )
                                                    
                                                    # Use the translated output instead
                                                    output_tokens = translate_output
                                                    
                                                    if st.session_state.debug_mode:
                                                        st.sidebar.info(f"Used two-step transcription: Speech→English→{transcription_language}")
                                                else:
                                                    # If it's another kind of error, re-raise
                                                    raise lang_error
                                            
                                            # Debug info if enabled
                                            if st.session_state.transformers_debug:
                                                st.sidebar.write(f"Output type: {type(output_tokens)}")
                                                st.sidebar.write(f"Output attributes: {dir(output_tokens)[:10]}...")
                                                if hasattr(output_tokens, 'shape'):
                                                    st.sidebar.write(f"Output shape: {output_tokens.shape}")
                                                elif isinstance(output_tokens, list):
                                                    st.sidebar.write(f"Output list length: {len(output_tokens)}")
                                            
                                            # Decode the output based on its type
                                            if hasattr(output_tokens, 'text_ids'):
                                                # Direct access to text_ids in the returned object
                                                segment_text = processor.decode(output_tokens.text_ids[0], skip_special_tokens=True)
                                            elif hasattr(output_tokens, 'sequences'):
                                                # For GenerateEncoderDecoderOutput objects
                                                segment_text = processor.batch_decode(output_tokens.sequences, skip_special_tokens=True)[0]
                                            elif isinstance(output_tokens, list):
                                                # For list output (batch of token sequences)
                                                segment_text = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
                                            elif hasattr(output_tokens, 'shape'):
                                                # For tensor output
                                                segment_text = processor.decode(output_tokens[0], skip_special_tokens=True)
                                            else:
                                                # If we can't decode it properly, convert to string but handle GenerateEncoderDecoderOutput
                                                if str(type(output_tokens)).find('GenerateEncoderDecoderOutput') >= 0:
                                                    # Handle GenerateEncoderDecoderOutput specifically
                                                    try:
                                                        segment_text = processor.batch_decode(output_tokens.sequences, skip_special_tokens=True)[0]
                                                    except Exception:
                                                        segment_text = "[Transcription error: unable to decode output]"
                                                else:
                                                    segment_text = "[Transcription error: unknown output format]"
                                            
                                            # For transformers Seamless, try to detect the language or use the specified one
                                            detected_language = None
                                            if hasattr(output_tokens, 'src_lang') and output_tokens.src_lang:
                                                # Map language code to full name
                                                language_map = {
                                                    "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German", 
                                                    "ita": "Italian", "cmn": "Chinese", "arb": "Arabic", "hin": "Hindi",
                                                    "rus": "Russian", "tur": "Turkish"
                                                }
                                                detected_language = language_map.get(output_tokens.src_lang, output_tokens.src_lang.title())
                                            elif selected_source_code:
                                                # If we specified a source language, use that
                                                language_map = {
                                                    "eng": "English", "spa": "Spanish", "fra": "French", "deu": "German", 
                                                    "ita": "Italian", "cmn": "Chinese", "arb": "Arabic", "hin": "Hindi",
                                                    "rus": "Russian", "tur": "Turkish"
                                                }
                                                detected_language = language_map.get(selected_source_code, source_language)
                                            else:
                                                # Fallback to the source language from UI selection
                                                detected_language = source_language if source_language != "Auto-detect" else "English"
                                            
                                        except Exception as gen_error:
                                            # Show detailed debug info if enabled
                                            if st.session_state.transformers_debug:
                                                st.sidebar.error(f"Error in generation: {gen_error}")
                                                import traceback
                                                st.sidebar.code(traceback.format_exc())
                                                st.sidebar.write("Trying alternative approach...")
                                                
                                                # Try direct text generation
                                                try:
                                                    # Try to get text directly using text-only generation
                                                    text_output = model.generate(**inputs, tgt_lang=selected_language_code, tasks=["asr"])
                                                    st.sidebar.write(f"Alt output: {text_output}")
                                                    if hasattr(text_output, 'text'):
                                                        segment_text = text_output.text[0] 
                                                    else:
                                                        segment_text = ""
                                                except Exception as alt_error:
                                                    st.sidebar.error(f"Alt approach failed: {alt_error}")
                                                    segment_text = ""
                                            else:
                                                # Just show a brief warning for users
                                                st.warning(f"Error in Seamless M4T generation: {gen_error}")
                                                segment_text = ""
                                            detected_language = None
                                    
                                except Exception as e:
                                    st.warning(f"Transformers Seamless model error: {e}")
                                    segment_text = ""
                                    detected_language = None
                            else:
                                segment_text = ""
                                detected_language = None
                        else:
                            # Unknown model type
                            st.warning(f"Unsupported model type: {model_type}")
                            segment_text = ""
                            detected_language = None
                        
                        # Only add if there's text
                        if segment_text.strip():
                            # Safety check - don't add raw tensor data to transcript
                            if 'tensor(' in segment_text or 'device=' in segment_text:
                                # Detected tensor data in output, use a placeholder
                                segment_text = "[Transcription error: received tensor data]"
                            
                            # Determine the language for this segment
                            segment_language = None
                            
                            # First check if we got a detected language from the transcription process
                            if detected_language is not None:
                                # Use the language detected by the transcription engine
                                segment_language = detected_language
                                print(f"Debug - Using detected language from transcription: {segment_language}")
                            # Otherwise use provided language settings
                            elif model_type == "whisper" and language and language != "Auto-detect":
                                segment_language = language
                                print(f"Debug - Using provided language: {segment_language}")
                            # For Seamless, use specified source language
                            elif model_type == "seamless" or model_type == "transformers_seamless":
                                if source_language != "Auto-detect":
                                    segment_language = source_language
                                    print(f"Debug - Using source language: {segment_language}")
                                else:
                                    segment_language = "English"  # Default when auto-detect
                                    print(f"Debug - Using default English for auto-detect")
                            # For Wav2Vec2, always English
                            elif model_type == "wav2vec2":
                                segment_language = "English"
                                print(f"Debug - Using default English for Wav2Vec2")
                            # Default fallback
                            else:
                                segment_language = "English"
                                print(f"Debug - Using default English (fallback)")
                            
                            # Store in transcript
                            full_transcript.append({
                                "speaker": speaker,
                                "start": start_time,
                                "end": end_time,
                                "text": segment_text.strip(),
                                "language": segment_language
                            })
                    except Exception as e:
                        error_count += 1
                        st.error(f"Error transcribing segment {segment_count}: {e}")
                        traceback.print_exc()
                
                # Clear the progress indicators
                progress_bar.empty()
                status_text.empty()
                
                if error_count > 0:
                    st.warning(f"Encountered {error_count} errors during transcription. Some segments may be missing.")
                
                if st.session_state.debug_mode:
                    st.sidebar.success(f"✓ Transcribed {len(full_transcript)} segments")
                
                # Format and display the transcript
                if full_transcript:
                    # Sort by start time
                    full_transcript.sort(key=lambda x: x["start"])
                    
                    # Store the transcript data in session state for later use
                    st.session_state.transcript_data = full_transcript
                    
                    # Extract speaker samples for identification
                    if not st.session_state.audio_samples:
                        speaker_samples = {}
                        for speaker in set(entry["speaker"] for entry in full_transcript):
                            # Find a good sample for this speaker (longer than 5 seconds ideally)
                            samples = [entry for entry in full_transcript if entry["speaker"] == speaker and entry["end"] - entry["start"] >= 3]
                            if samples:
                                # Use the longest sample
                                sample = max(samples, key=lambda x: x["end"] - x["start"])
                            else:
                                # If no good sample, just use the first one we find
                                samples = [entry for entry in full_transcript if entry["speaker"] == speaker]
                                if samples:
                                    sample = samples[0]
                                else:
                                    continue
                            
                            # Store the sample start/end times
                            start_time = sample["start"]
                            # Use at most 5 seconds
                            end_time = min(start_time + 5, sample["end"])
                            
                            # Extract audio segment
                            try:
                                segment_audio = audio_array[int(start_time * sample_rate):int(end_time * sample_rate)]
                                # Create a temporary WAV file for the segment
                                segment_filename = f"speaker_{speaker}_{uuid.uuid4().hex}.wav"
                                segment_filepath = os.path.join(tempfile.gettempdir(), segment_filename)
                                
                                # Write the audio segment to a file
                                sf.write(segment_filepath, segment_audio, sample_rate)
                                
                                # Read the file back to get binary data
                                with open(segment_filepath, 'rb') as f:
                                    audio_data = f.read()
                                
                                # Store the sample
                                speaker_samples[speaker] = {
                                    "audio": audio_data,
                                    "start_time": start_time,
                                    "end_time": end_time,
                                    "sample_text": sample["text"]
                                }
                                
                                # Clean up
                                safe_remove_file(segment_filepath)
                            except Exception as e:
                                if st.session_state.debug_mode:
                                    st.sidebar.warning(f"Failed to extract audio sample for Speaker {speaker}: {e}")
                    
                    # Store samples in session state
                    st.session_state.audio_samples = speaker_samples
                else:
                    st.error("No transcribable speech segments were detected. Please check your audio file or try adjusting the minimum segment duration in Advanced Options.")
            
            except Exception as e:
                st.error(f"Error processing the audio: {str(e)}")
                st.exception(e)
            
            finally:
                # Clean up the temporary file safely
                if clean_temp_files:
                    safe_remove_file(temp_file_path)

    # Always display the transcript and speaker identification UI when we have transcript data
    if st.session_state.transcript_data:
        full_transcript = st.session_state.transcript_data
        
        # Add visual separation before speaker identification
        st.divider()
        
        # Add Speaker Identification form with header
        st.header("Speaker Identification")
        speaker_id_expander = st.expander("", expanded=True)
        with speaker_id_expander:
            st.markdown("*Use this form to identify speakers in the audio. You must save all speakers before you can update the transcript with their names.*")
            
            # Create columns for the update button
            update_col1, update_col2, update_col3 = st.columns([2, 1, 1])
            with update_col2:
                update_button = st.button("Update Transcript", key="update_transcript", disabled=len(st.session_state.speaker_names) == 0)
            with update_col3:
                if st.button("Reset All Names", key="reset_speakers"):
                    st.session_state.speaker_names = {}
                    # No need to rerun - the UI will update on next render
            
            # Show a message about current identifications
            if st.session_state.speaker_names:
                identified_count = len(st.session_state.speaker_names)
                total_count = len(set(entry["speaker"] for entry in full_transcript))
                st.info(f"{identified_count} out of {total_count} speakers identified")
            
            st.write("Listen to each speaker and assign their names:")
            
            # Create a form for speaker identification using a container for better styling
            speaker_form = st.container(border=True)
            with speaker_form:
                for speaker in sorted(set(entry["speaker"] for entry in full_transcript)):
                    # Check if this speaker has a saved name
                    is_saved = speaker in st.session_state.speaker_names and st.session_state.speaker_names[speaker] != ""
                    
                    # Create a container for this speaker row
                    speaker_row = st.container()
                    
                    with speaker_row:
                        # Create columns for each speaker row
                        col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
                        
                        with col1:
                            # Add a visual indicator for saved speakers
                            if is_saved:
                                st.markdown(f"**Speaker {speaker}** ✓")
                            else:
                                st.write(f"Speaker {speaker}")
                                
                            if speaker in st.session_state.audio_samples and "sample_text" in st.session_state.audio_samples[speaker]:
                                sample_text = st.session_state.audio_samples[speaker]["sample_text"]
                                st.caption(f"Sample text: \"{sample_text[:50]}...\"" if len(sample_text) > 50 else f"Sample text: \"{sample_text}\"")
                        
                        with col2:
                            # Audio playback button
                            if speaker in st.session_state.audio_samples and "audio" in st.session_state.audio_samples[speaker]:
                                st.audio(st.session_state.audio_samples[speaker]["audio"], format="audio/wav")
                                if "start_time" in st.session_state.audio_samples[speaker]:
                                    st.caption(f"Sample from {st.session_state.audio_samples[speaker]['start_time']:.2f}s")
                            else:
                                st.write("No sample available")
                        
                        with col3:
                            # Text input for speaker name
                            speaker_name = st.text_input(
                                f"Name for Speaker {speaker}",
                                value=st.session_state.speaker_names.get(speaker, ""),
                                key=f"name_{speaker}",
                                label_visibility="collapsed",
                                placeholder="Enter speaker's name"
                            )
                        
                        with col4:
                            # Icon buttons for save/cancel
                            icon_cols = st.columns([1, 1])
                            with icon_cols[0]:
                                save_button = st.button("💾", key=f"save_{speaker}", help="Save name")
                                if save_button:
                                    if not speaker_name:
                                        st.warning(f"Please enter a name for Speaker {speaker}")
                                    else:
                                        st.session_state.speaker_names[speaker] = speaker_name
                                        # Update without rerunning
                            with icon_cols[1]:
                                if st.button("❌", key=f"cancel_{speaker}", help="Clear name"):
                                    if speaker in st.session_state.speaker_names:
                                        del st.session_state.speaker_names[speaker]
                                        # Update without rerunning
                    
                    # Add a small divider between speakers for better readability
                    if speaker != sorted(set(entry["speaker"] for entry in full_transcript))[-1]:
                        st.divider()
        
        # Add visual separation before transcript
        st.divider()
        
        # Check for non-preferred language segments
        non_preferred_segments = []
        for i, entry in enumerate(full_transcript):
            if entry["language"] != st.session_state.preferred_language and entry["language"] is not None:
                entry_copy = entry.copy()
                entry_copy["index"] = i
                non_preferred_segments.append(entry_copy)
        
        # If there are non-preferred language segments, show translation options
        if non_preferred_segments:
            # Add Translation section with header
            st.header("Translation")
            translation_expander = st.expander("", expanded=True)
            with translation_expander:
                st.markdown(f"*{len(non_preferred_segments)} segments detected in languages other than {st.session_state.preferred_language}*")
                
                # Get system platform to determine available translation models
                system_platform = platform.system().lower()
                
                # Automatically select translation method based on target language and platform
                # - Whisper for English (fast but English-only target)
                # - Seamless M4T on Linux or NLLB on Windows/macOS (for all language pairs)
                if st.session_state.preferred_language == "English":
                    translation_method = "Whisper (Fast, English only)"
                    default_target = "English"
                else:
                    if system_platform == "linux":
                        translation_method = "Seamless M4T (Multi-language)"
                    else:
                        translation_method = "NLLB (Multi-language)"
                    default_target = st.session_state.preferred_language
                
                # Add target language selection
                target_language = st.selectbox(
                    "Translate to:", 
                    sorted(preferred_languages),
                    index=sorted(preferred_languages).index(default_target) if default_target in preferred_languages else 0,
                    key="target_language"
                )
                
                # If Whisper is selected and target is not English, switch to appropriate multi-language model
                if "Whisper" in translation_method and target_language != "English":
                    if system_platform == "linux":
                        translation_method = "Seamless M4T (Multi-language)"
                        st.info("Switched to Seamless M4T for non-English target language")
                    else:
                        translation_method = "NLLB (Multi-language)"
                        st.info("Switched to NLLB for non-English target language")
                
                # Add model explanation based on the selected method
                if "Whisper" in translation_method:
                    st.caption("Whisper provides fast translation but only supports English as a target language")
                elif "Seamless" in translation_method:
                    st.caption("Seamless M4T is a powerful multilingual model that supports many language pairs")
                elif "NLLB" in translation_method:
                    st.caption("NLLB (No Language Left Behind) supports translation between 200+ languages")
                
                # Change the translate all button text
                translate_button_text = f"Translate All Segments to {target_language}"
                translate_all = st.button(translate_button_text)
                
                # Individual translation options - use a container instead of an expander
                st.markdown("### Translate Individual Segments")
                individual_segments_container = st.container(border=True)
                with individual_segments_container:
                    for segment in non_preferred_segments:
                        speaker_display = f"Speaker {segment['speaker']}"
                        if segment['speaker'] in st.session_state.speaker_names:
                            speaker_display = st.session_state.speaker_names[segment['speaker']]
                        
                        segment_key = f"{segment['start']}_{segment['end']}_{segment['speaker']}"
                        
                        # Check if this segment is already translated
                        already_translated = (segment_key in st.session_state.translated_segments and 
                                             st.session_state.translated_segments[segment_key]["translated_text"] is not None)
                        
                        # If translated, show the current target language
                        current_target = None
                        if already_translated:
                            current_target = st.session_state.translated_segments[segment_key]["target_language"]
                        
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{speaker_display}** ({segment['language']}): {segment['text']}")
                            if already_translated:
                                st.caption(f"Currently translated to: {current_target}")
                        with col2:
                            button_text = "Translate" if not already_translated else "Retranslate"
                            
                            # Add "to Language" indication if changing languages
                            if already_translated and current_target != target_language:
                                button_text = f"Retranslate to {target_language}"
                            
                            if st.button(button_text, key=f"translate_{segment_key}"):
                                # Always mark for translation, even if already translated
                                st.session_state.translated_segments[segment_key] = {
                                    "original": segment,
                                    "translated_text": None,  # Setting to None marks it for translation
                                    "target_language": target_language,
                                    "method": translation_method
                                }
                                # We'll process translations at the end rather than rerunning immediately
                
            # Modify the translate all button handler to process all segments in the transcript
            if translate_all:
                # Process ALL segments, not just non-preferred ones
                for entry in full_transcript:
                    segment_key = f"{entry['start']}_{entry['end']}_{entry['speaker']}"
                    
                    # Skip segments that are already in the target language
                    if entry["language"] == target_language:
                        continue
                    
                    # Determine which translation method to use based on target language and platform
                    if target_language == "English":
                        selected_method = "Whisper (Fast, English only)"
                    else:
                        if system_platform == "linux":
                            selected_method = "Seamless M4T (Multi-language)"
                        else:
                            selected_method = "NLLB (Multi-language)"
                    
                    # Check if we need to retranslate (new target language)
                    needs_retranslation = True
                    if segment_key in st.session_state.translated_segments:
                        current_info = st.session_state.translated_segments[segment_key]
                        # Only mark for retranslation if target language changed or not already translated
                        if (current_info["translated_text"] is not None and 
                            current_info["target_language"] == target_language and
                            current_info["method"] == selected_method):
                            needs_retranslation = False
                    
                    if needs_retranslation:
                        # Create a proper segment object similar to what's in non_preferred_segments
                        segment_obj = entry.copy()
                        segment_obj["index"] = full_transcript.index(entry)
                        
                        st.session_state.translated_segments[segment_key] = {
                            "original": segment_obj,
                            "translated_text": None,
                            "target_language": target_language,
                            "method": selected_method
                        }
                
                # Only trigger rerun if there are segments to translate
                pending_count = sum(1 for v in st.session_state.translated_segments.values() if v["translated_text"] is None)
                # We'll process translations immediately instead of doing a rerun
            
            # Process any pending translations
            translation_performed = False
            
            if st.session_state.translated_segments:
                # Process pending translations
                pending_translations = [k for k, v in st.session_state.translated_segments.items() if v["translated_text"] is None]
                
                if pending_translations:
                    # Group translations by method
                    whisper_translations = []
                    seamless_translations = []
                    nllb_translations = []
                    
                    for segment_key in pending_translations:
                        segment_info = st.session_state.translated_segments[segment_key]
                        method = segment_info.get("method", "")
                        
                        # Sort translations into appropriate groups based on method
                        if "Whisper" in method:
                            whisper_translations.append(segment_key)
                        elif "Seamless" in method:
                            seamless_translations.append(segment_key)
                        elif "NLLB" in method:
                            nllb_translations.append(segment_key)
                        else:
                            # For backward compatibility or unspecified methods
                            if system_platform == "linux":
                                seamless_translations.append(segment_key)
                            else:
                                nllb_translations.append(segment_key)
                    
                    # Process Whisper translations first (faster)
                    if whisper_translations:
                        with st.spinner(f"Translating {len(whisper_translations)} segments with Whisper..."):
                            # Find the whisper model - either use the one that's already loaded or load a new one
                            whisper_model = None
                            if 'transcription_model' in locals() and model_type == "whisper":
                                whisper_model = transcription_model
                            else:
                                # Load a small Whisper model for translation
                                whisper_model = load_whisper_model("small", use_gpu)
                            
                            progress_bar = st.progress(0)
                            
                            for i, segment_key in enumerate(whisper_translations):
                                segment_info = st.session_state.translated_segments[segment_key]
                                original_segment = segment_info["original"]
                                
                                # Use the start and end times to extract the segment
                                start_time = original_segment["start"]
                                end_time = original_segment["end"]
                                
                                try:
                                    # Directly translate using Whisper's built-in translation
                                    # Get audio data from session state
                                    if st.session_state.audio_data is None:
                                        st.error("Audio data is not available for translation. Please reload the file.")
                                        continue
                                        
                                    audio_array = st.session_state.audio_data["array"]
                                    sample_rate = st.session_state.audio_data["sample_rate"]
                                    
                                    # Create a temp file for this specific segment
                                    segment_audio = audio_array[int(start_time * sample_rate):int(end_time * sample_rate)]
                                    segment_filename = f"whisper_translate_{uuid.uuid4().hex}.wav"
                                    segment_filepath = os.path.join(tempfile.gettempdir(), segment_filename)
                                    
                                    # Write the audio segment to a file
                                    sf.write(segment_filepath, segment_audio, sample_rate)
                                    
                                    # Get the source language code for better translation
                                    source_lang_code = get_language_code(original_segment["language"])
                                    
                                    # Translate with Whisper
                                    translated_text, _ = transcribe_whisper_segment(
                                        segment_filepath,
                                        0,  # Start from the beginning of this segment file
                                        len(segment_audio) / sample_rate,  # Duration of the segment
                                        whisper_model,
                                        language=source_lang_code,  # Provide source language code for better translation
                                        task="translate"  # Use Whisper's translate task
                                    )
                                    
                                    # Clean up the temporary file
                                    safe_remove_file(segment_filepath)
                                    
                                    # Store the translated text
                                    st.session_state.translated_segments[segment_key]["translated_text"] = translated_text
                                    st.session_state.translated_segments[segment_key]["target_language"] = target_language  # Use the selected target language
                                    translation_performed = True
                                except Exception as e:
                                    st.error(f"Whisper translation error: {e}")
                                    # Set a placeholder for failed translations
                                    st.session_state.translated_segments[segment_key]["translated_text"] = f"[Translation failed: {e}]"
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(whisper_translations))
                            
                            # Clear progress when done
                            progress_bar.empty()
                            if translation_performed and whisper_translations:
                                st.success("Whisper translation completed!")
                    
                    # Process translations with Seamless M4T on Linux
                    if seamless_translations and system_platform == "linux":
                        with st.spinner(f"Translating {len(seamless_translations)} segments with Seamless M4T..."):
                            try:
                                from seamless_communication.inference import Translator
                                
                                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                                torch_dtype = torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32
                                
                                translator = Translator(
                                    "facebook/seamless-m4t-v2-large",
                                    "vocoder_36langs",
                                    device,
                                    torch_dtype
                                )
                                
                                # Language code mapping for Seamless M4T
                                language_map = {
                                    "English": "eng", "Spanish": "spa", "French": "fra", "German": "deu", 
                                    "Italian": "ita", "Portuguese": "por", "Chinese": "cmn", "Japanese": "jpn", 
                                    "Korean": "kor", "Arabic": "arb", "Russian": "rus"
                                }
                                
                                progress_bar = st.progress(0)
                                
                                for i, segment_key in enumerate(seamless_translations):
                                    segment_info = st.session_state.translated_segments[segment_key]
                                    original_segment = segment_info["original"]
                                    source_text = original_segment["text"]
                                    target_lang_code = language_map.get(segment_info["target_language"], "eng")
                                    
                                    # Detect source language
                                    source_lang_code = language_map.get(original_segment["language"], None)
                                    
                                    try:
                                        # Translate the text using Seamless M4T
                                        result = translator.predict(
                                            source_text,
                                            "T2TT",  # Text to text translation
                                            tgt_lang=target_lang_code,
                                            src_lang=source_lang_code
                                        )
                                        
                                        # Store the translated text
                                        st.session_state.translated_segments[segment_key]["translated_text"] = result.text
                                        translation_performed = True
                                    except Exception as e:
                                        st.error(f"Seamless M4T translation error: {e}")
                                        # Set a placeholder for failed translations
                                        st.session_state.translated_segments[segment_key]["translated_text"] = f"[Translation failed: {e}]"
                                    
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(seamless_translations))
                                
                                # Clear progress when done
                                progress_bar.empty()
                                if translation_performed and seamless_translations:
                                    st.success("Seamless M4T translation completed!")
                                    
                            except ImportError:
                                st.error("""
                                Seamless Communication package is not available for translation.
                                Please run the installation script:
                                ```
                                ./run.sh
                                ```
                                """)
                    
                    # Process translations with NLLB on Windows/macOS
                    if nllb_translations and system_platform != "linux":
                        with st.spinner(f"Translating {len(nllb_translations)} segments with NLLB..."):
                            try:
                                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
                                
                                # Use NLLB 200-language model for translation
                                model_name = "facebook/nllb-200-distilled-600M"
                                
                                tokenizer = AutoTokenizer.from_pretrained(model_name)
                                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                                
                                if use_gpu and torch.cuda.is_available():
                                    model = model.to("cuda")
                                
                                # NLLB language code mapping (different from Seamless)
                                language_map = {
                                    "English": "eng_Latn", "Spanish": "spa_Latn", "French": "fra_Latn", 
                                    "German": "deu_Latn", "Italian": "ita_Latn", "Portuguese": "por_Latn", 
                                    "Chinese": "zho_Hans", "Japanese": "jpn_Jpan", "Korean": "kor_Hang", 
                                    "Arabic": "ara_Arab", "Russian": "rus_Cyrl", "Turkish": "tur_Latn",
                                    "Dutch": "nld_Latn", "Polish": "pol_Latn", "Romanian": "ron_Latn",
                                    "Ukrainian": "ukr_Cyrl", "Czech": "ces_Latn", "Hungarian": "hun_Latn",
                                    "Finnish": "fin_Latn", "Swedish": "swe_Latn", "Greek": "ell_Grek",
                                    "Bulgarian": "bul_Cyrl", "Danish": "dan_Latn", "Norwegian": "nob_Latn",
                                    "Catalan": "cat_Latn", "Hebrew": "heb_Hebr", "Indonesian": "ind_Latn",
                                    "Thai": "tha_Thai", "Vietnamese": "vie_Latn", "Hindi": "hin_Deva",
                                    "Bengali": "ben_Beng", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
                                    "Urdu": "urd_Arab", "Swahili": "swh_Latn", "Welsh": "cym_Latn"
                                }
                                
                                # Create translation pipeline
                                translator = pipeline("translation", model=model, tokenizer=tokenizer)
                                
                                progress_bar = st.progress(0)
                                
                                for i, segment_key in enumerate(nllb_translations):
                                    segment_info = st.session_state.translated_segments[segment_key]
                                    original_segment = segment_info["original"]
                                    source_text = original_segment["text"]
                                    target_lang_code = language_map.get(segment_info["target_language"], "eng_Latn")
                                    
                                    # Detect source language
                                    source_lang_code = language_map.get(original_segment["language"], "eng_Latn")
                                    
                                    try:
                                        # Translate the text using NLLB model
                                        translation = translator(
                                            source_text, 
                                            src_lang=source_lang_code, 
                                            tgt_lang=target_lang_code,
                                            max_length=512
                                        )
                                        
                                        # Get translated text from the result
                                        translated_text = translation[0]["translation_text"]
                                        
                                        # Store the translated text
                                        st.session_state.translated_segments[segment_key]["translated_text"] = translated_text
                                        translation_performed = True
                                    except Exception as e:
                                        st.error(f"NLLB translation error: {e}")
                                        # Set a placeholder for failed translations
                                        st.session_state.translated_segments[segment_key]["translated_text"] = f"[Translation failed: {e}]"
                                    
                                    # Update progress
                                    progress_bar.progress((i + 1) / len(nllb_translations))
                                
                                # Clear progress when done
                                progress_bar.empty()
                                if translation_performed and nllb_translations:
                                    st.success("NLLB translation completed!")
                                    
                            except ImportError:
                                st.error("""
                                Transformers library is not properly installed for translation.
                                Please run the installation script:
                                ```
                                run.bat  # On Windows
                                ./run.sh  # On macOS
                                ```
                                """)
                    
                    # Check if any translations are left from the original allocation
                    # This can happen if the platform doesn't match the expected one
                    remaining_translations = [k for k, v in st.session_state.translated_segments.items() 
                                             if v["translated_text"] is None and k in pending_translations]
                    
                    if remaining_translations:
                        st.warning(f"{len(remaining_translations)} segments could not be translated with the selected method. Please try a different method.")
            
            # If translations were performed, trigger a rerun to update the display
            if translation_performed:
                st.rerun()
            
            # Show translation results if available
            if any(v["translated_text"] is not None for v in st.session_state.translated_segments.values()):
                # Translation Results section - collapsed by default with header
                st.header("Translation Results")
                translation_results_expander = st.expander("", expanded=False)
                with translation_results_expander:
                    # Option to clear all translations
                    if st.button("Clear All Translations"):
                        st.session_state.translated_segments = {}
                        st.rerun()
                    
                    # Display translations
                    for segment_key, segment_info in st.session_state.translated_segments.items():
                        if segment_info["translated_text"] is not None:
                            original_segment = segment_info["original"]
                            speaker_display = f"Speaker {original_segment['speaker']}"
                            if original_segment['speaker'] in st.session_state.speaker_names:
                                speaker_display = st.session_state.speaker_names[original_segment['speaker']]
                            
                            st.markdown(f"**{speaker_display}** ({original_segment['language']} → {segment_info['target_language']}):")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.text_area("Original", original_segment["text"], height=100, key=f"orig_{segment_key}", disabled=True)
                            with col2:
                                st.text_area("Translation", segment_info["translated_text"], height=100, key=f"trans_{segment_key}", disabled=True)
            
            # Add visual separator before transcript
            st.divider()
        
        # Create updated transcript with translations if available
        if st.session_state.translated_segments:
            # Apply speaker names and translations to transcript
            updated_transcript = []
            for i, entry in enumerate(full_transcript):
                entry_copy = entry.copy()
                
                # Apply speaker names if available
                speaker_id = entry["speaker"]
                if speaker_id in st.session_state.speaker_names:
                    entry_copy["display_name"] = st.session_state.speaker_names[speaker_id]
                else:
                    entry_copy["display_name"] = f"Speaker {speaker_id}"
                
                # Apply translations if available
                segment_key = f"{entry['start']}_{entry['end']}_{entry['speaker']}"
                if segment_key in st.session_state.translated_segments and st.session_state.translated_segments[segment_key]["translated_text"] is not None:
                    # Add translation to the entry
                    translation_info = st.session_state.translated_segments[segment_key]
                    entry_copy["translated_text"] = translation_info["translated_text"]
                    entry_copy["target_language"] = translation_info["target_language"]
                
                updated_transcript.append(entry_copy)
            
            # Format the updated transcript with translations
            formatted_text = ""
            for entry in updated_transcript:
                display_name = entry.get("display_name", f"Speaker {entry['speaker']}")
                # Add language info and original text
                formatted_text += f"[{entry['start']:.2f} - {entry['end']:.2f}] - {entry['language']} - {display_name}:\n"
                formatted_text += f"{entry['text']}\n"
                
                # Add translation if available
                if "translated_text" in entry:
                    formatted_text += f"[Translation to {entry['target_language']}]: {entry['translated_text']}\n"
                
                formatted_text += "\n"
        elif update_button and st.session_state.speaker_names:
            # Apply speaker names to transcript
            updated_transcript = []
            for entry in full_transcript:
                speaker_id = entry["speaker"]
                if speaker_id in st.session_state.speaker_names:
                    entry_copy = entry.copy()
                    entry_copy["display_name"] = st.session_state.speaker_names[speaker_id]
                    updated_transcript.append(entry_copy)
                else:
                    entry_copy = entry.copy()
                    entry_copy["display_name"] = f"Speaker {speaker_id}"
                    updated_transcript.append(entry_copy)
            
            # Format the updated transcript
            formatted_text = ""
            for entry in updated_transcript:
                display_name = entry.get("display_name", f"Speaker {entry['speaker']}")
                formatted_text += f"[{entry['start']:.2f} - {entry['end']:.2f}] - {entry['language']} - {display_name}:\n"
                formatted_text += f"{entry['text']}\n\n"
        else:
            # Format the original transcript
            formatted_text = ""
            for entry in full_transcript:
                formatted_text += f"[{entry['start']:.2f} - {entry['end']:.2f}] - {entry['language']} - Speaker {entry['speaker']}:\n"
                formatted_text += f"{entry['text']}\n\n"
        
        # Transcript section with header
        st.header("Transcript")
        transcript_expander = st.expander("", expanded=True)
        with transcript_expander:
            st.text_area(label="Complete Transcript", value=formatted_text, height=400)
            
            # Update download options to include speaker names and translations
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(label="Download as Text", data=formatted_text, file_name="transcript.txt")
            
            # Also offer SRT format
            with col2:
                srt_text = ""
                for i, entry in enumerate(updated_transcript if (update_button and st.session_state.speaker_names) or st.session_state.translated_segments else full_transcript, 1):
                    start_time_str = f"{int(entry['start'] // 3600):02d}:{int((entry['start'] % 3600) // 60):02d}:{int(entry['start'] % 60):02d},{int((entry['start'] % 1) * 1000):03d}"
                    end_time_str = f"{int(entry['end'] // 3600):02d}:{int((entry['end'] % 3600) // 60):02d}:{int(entry['end'] % 60):02d},{int((entry['end'] % 1) * 1000):03d}"
                    
                    # Use display_name if available, otherwise use the default speaker ID
                    display_name = entry.get("display_name", f"Speaker {entry['speaker']}")
                    
                    # Base SRT entry
                    srt_text += f"{i}\n{start_time_str} --> {end_time_str}\n{entry['language']} - {display_name}: {entry['text']}"
                    
                    # Add translation if available
                    if "translated_text" in entry:
                        srt_text += f"\n[{entry['target_language']}]: {entry['translated_text']}"
                    
                    srt_text += "\n\n"
                
                st.download_button(label="Download as SRT", data=srt_text, file_name="transcript.srt")