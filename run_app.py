#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Streamlit CLASSIFIT - AI Image Classification
Mendukung mode demo ketika TensorFlow tidak tersedia
"""

import subprocess
import sys
import os

def check_core_dependencies():
    """Check if core dependencies are installed"""
    missing_deps = []
    
    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import PIL
    except ImportError:
        missing_deps.append("pillow")
    
    return missing_deps

def check_ml_dependencies():
    """Check if ML dependencies are available (optional)"""
    ml_status = {
        'tflite_runtime': False,
        'tensorflow': False,
        'model_file': False
    }
    
    # Check tflite-runtime
    try:
        import tflite_runtime
        ml_status['tflite_runtime'] = True
    except ImportError:
        pass
    
    # Check tensorflow
    try:
        import tensorflow
        ml_status['tensorflow'] = True
    except ImportError:
        pass
    
    # Check model file
    ml_status['model_file'] = os.path.exists(os.path.join("tflite", "model.tflite"))
    
    return ml_status

def print_status():
    """Print current system status"""
    print("🔍 Checking system status...")
    print("-" * 40)
    
    # Check core dependencies
    missing_core = check_core_dependencies()
    if missing_core:
        print(f"❌ Missing core dependencies: {', '.join(missing_core)}")
        return False
    else:
        print("✅ Core dependencies: OK")
    
    # Check ML dependencies (optional)
    ml_status = check_ml_dependencies()
    
    if ml_status['tflite_runtime']:
        print("✅ TensorFlow Lite Runtime: Available")
    elif ml_status['tensorflow']:
        print("✅ TensorFlow: Available")
    else:
        print("⚠️  ML Libraries: Not available (will run in demo mode)")
    
    if ml_status['model_file']:
        print("✅ Model file: Found")
    else:
        print("⚠️  Model file: Not found (will use demo predictions)")
    
    # Overall status
    if any([ml_status['tflite_runtime'], ml_status['tensorflow']]) and ml_status['model_file']:
        print("🚀 Status: Ready for AI inference")
    else:
        print("🎭 Status: Demo mode ready")
    
    print("-" * 40)
    return True

def install_dependencies():
    """Install missing dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        print("🚀 Starting Streamlit application...")
        print("📝 The app will open in your browser")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the application")
        print("=" * 50)
        
        # Run streamlit with optimal configuration
        cmd = [
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    print("=" * 50)
    print("🎯 CLASSIFIT - AI Image Classification")
    print("🤖 Intelligent Computer Vision Platform")
    print("=" * 50)
    
    # Check system status
    if not print_status():
        print("\n📦 Attempting to install missing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies. Please run manually:")
            print("   pip install -r requirements.txt")
            return
        
        # Recheck after installation
        print("\n🔄 Rechecking dependencies...")
        if not check_core_dependencies():
            print_status()
    
    print("\n" + "=" * 50)
    print("🎬 LAUNCHING APPLICATION")
    print("=" * 50)
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main() 