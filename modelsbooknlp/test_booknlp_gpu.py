from booknlp.booknlp import BookNLP
import torch
import os

# Check GPU availability
def check_gpu_availability():
    """Check if GPU is available and configure accordingly"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"🔢 CUDA Version: {torch.version.cuda}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
        return True, device
    else:
        print("⚠️  GPU not available, using CPU")
        return False, None

# Configure GPU settings
gpu_available, device = check_gpu_availability()

# Enhanced model parameters with GPU support
model_params = {
    "pipeline": "entity,quote,supersense,event,coref",
    "model": "custom",
    "entity_model_path": r"C:\Users\Adhyan\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0_modified.model",
    "coref_model_path": r"C:\Users\Adhyan\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0_modified.model",
    "quote_attribution_model_path": r"C:\Users\Adhyan\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1_modified.model",
    "bert_model_path": r"C:\Users\Adhyan\.cache\huggingface\hub",
    
    # GPU Configuration
    "use_gpu": gpu_available,
    "gpu_device": device if gpu_available else None,
    "batch_size": 32 if gpu_available else 16,  # Larger batch size for GPU
    "max_length": 512,  # Optimize for GPU memory
}

# Set environment variables for GPU optimization
if gpu_available:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

print(f"🔧 Model Configuration:")
print(f"   GPU Enabled: {model_params['use_gpu']}")
print(f"   Batch Size: {model_params['batch_size']}")
print(f"   Max Length: {model_params['max_length']}")

# Initialize BookNLP with GPU support
booknlp = BookNLP("en", model_params)

# Input file (UTF-8 encoded)
input_file = "sample.txt"

# Output directory
output_directory = "output/"

# Book ID
book_id = "samples"

print(f"\n🚀 Starting BookNLP processing with GPU acceleration...")

# Process with timing
import time
start_time = time.time()

# Pass the filename, not the text content
booknlp.process(input_file, output_directory, book_id)

end_time = time.time()
processing_time = end_time - start_time

print(f"✅ Processing completed in {processing_time:.2f} seconds")
if gpu_available:
    print(f"🚀 GPU acceleration was used")
else:
    print(f"💻 CPU processing was used")
