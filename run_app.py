#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Streamlit Klasifikasi Citra Intel
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import tensorflow
        import PIL
        print("✅ Semua dependencies sudah terinstall")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("📦 Menjalankan: pip install -r requirements.txt")
        return False

def run_streamlit_app():
    """Run the Streamlit app"""
    try:
        print("🚀 Menjalankan aplikasi Streamlit...")
        print("📝 Aplikasi akan terbuka di browser Anda")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Tekan Ctrl+C untuk menghentikan aplikasi\n")
        
        # Run streamlit with optimal configuration
        cmd = [
            "python3", "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Aplikasi dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error menjalankan aplikasi: {e}")

def main():
    print("=" * 50)
    print("🧠 Klasifikasi Citra Intel - Streamlit App")
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists("saved_model"):
        print("❌ Model tidak ditemukan!")
        print("📁 Pastikan folder 'saved_model' ada di direktori ini")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("\n📦 Installing dependencies...")
        try:
            subprocess.run(["python3", "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies berhasil diinstall")
        except subprocess.CalledProcessError:
            print("❌ Gagal install dependencies")
            return
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main() 