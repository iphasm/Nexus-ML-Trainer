"""
Health Check Wrapper for Railway
Runs the training script while providing a /health endpoint
"""

import os
import sys
import threading
import subprocess
from flask import Flask

app = Flask(__name__)

# Track if training is running
training_thread = None
training_complete = False
training_error = None


@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    if training_error:
        return {"status": "error", "message": str(training_error)}, 500
    elif training_complete:
        return {"status": "completed", "message": "Training finished successfully"}, 200
    else:
        return {"status": "running", "message": "Training in progress"}, 200


@app.route('/')
def index():
    """Root endpoint"""
    return {"service": "ML Cloud Trainer", "status": "active"}


def run_training():
    """Run the training script"""
    global training_complete, training_error
    try:
        # Get training arguments from environment or use defaults
        candles = os.getenv('TRAINING_CANDLES', '35000')
        
        result = subprocess.run(
            [sys.executable, 'train_cortex.py', '--candles', candles, '--auto'],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            training_error = f"Training exited with code {result.returncode}"
        else:
            training_complete = True
            
    except Exception as e:
        training_error = str(e)


if __name__ == '__main__':
    # Start training in background thread
    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    
    # Run Flask server (Railway will use PORT env var)
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
