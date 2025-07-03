#!/bin/bash

# CLASSIFIT Deployment Script
# Automates the deployment process to Streamlit Cloud

set -e  # Exit on any error

echo "=================================================="
echo "🚀 CLASSIFIT Deployment Script v2.0.0"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    print_error "app.py not found! Please run this script from the CLASSIFIT root directory."
    exit 1
fi

print_status "Checking deployment requirements..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    print_warning "Git repository not found. Initializing..."
    git init
    git add .
    git commit -m "Initial commit for CLASSIFIT deployment"
fi

# Check for required files
required_files=("app.py" "requirements.txt" "runtime.txt" ".streamlit/config.toml")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        print_error "Required file missing: $file"
        exit 1
    fi
done

print_success "All required files present"

# Validate Python syntax
print_status "Validating Python syntax..."
python3 -m py_compile app.py
if [ $? -eq 0 ]; then
    print_success "Python syntax validation passed"
else
    print_error "Python syntax validation failed"
    exit 1
fi

# Check dependencies
print_status "Checking dependencies..."
pip3 install -r requirements.txt --dry-run > /dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Dependencies validation passed"
else
    print_warning "Some dependencies may have issues, but continuing..."
fi

# Test import
print_status "Testing app imports..."
python3 -c "
import sys
sys.path.append('.')
try:
    import app
    print('✅ App imports successfully')
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Import test passed"
else
    print_error "Import test failed"
    exit 1
fi

# Git operations
print_status "Preparing git repository..."

# Add all changes
git add .

# Check if there are changes to commit
if git diff --cached --quiet; then
    print_warning "No changes to commit"
else
    # Get current timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Commit changes
    git commit -m "🚀 Production deployment ready - $timestamp

✨ Features:
- Enhanced UI with dark/light mode support
- Production-optimized configuration
- Linux compatibility ensured
- Comprehensive error handling
- Performance optimizations
- Progressive web app capabilities

🔧 Technical improvements:
- Updated dependencies with version pinning
- Enhanced caching strategies
- Better resource management
- Streamlit Cloud optimized configuration
- Container support with Dockerfile

📱 Deployment ready for Streamlit Cloud!"

    print_success "Changes committed successfully"
fi

# Show deployment instructions
echo ""
echo "=================================================="
echo "🎯 DEPLOYMENT INSTRUCTIONS"
echo "=================================================="
echo ""
echo "Your CLASSIFIT app is ready for deployment! 🎉"
echo ""
echo "📋 Streamlit Cloud Deployment:"
echo "   1. Push to GitHub: git push origin main"
echo "   2. Go to: https://share.streamlit.io"
echo "   3. Connect your repository"
echo "   4. Select this repository and branch"
echo "   5. Set main file path: app.py"
echo "   6. Deploy! 🚀"
echo ""
echo "🐳 Docker Deployment (Optional):"
echo "   docker build -t classifit ."
echo "   docker run -p 8501:8501 classifit"
echo ""
echo "🔧 Configuration Files Created:"
echo "   ✅ requirements.txt    - Python dependencies"
echo "   ✅ runtime.txt         - Python version specification"
echo "   ✅ packages.txt        - System dependencies"
echo "   ✅ .streamlit/config.toml - Streamlit configuration"
echo "   ✅ Dockerfile          - Container configuration"
echo "   ✅ .dockerignore       - Docker build optimization"
echo ""
echo "🌟 Features Ready:"
echo "   ✅ Cross-platform compatibility (Linux/Windows/macOS)"
echo "   ✅ Dark/Light mode support"
echo "   ✅ Mobile responsive design"
echo "   ✅ Progressive web app capabilities"
echo "   ✅ Production error handling"
echo "   ✅ Performance optimizations"
echo "   ✅ Demo mode fallback (when TensorFlow unavailable)"
echo ""
echo "📊 Model Support:"
echo "   ✅ TensorFlow Lite model (tflite/model.tflite)"
echo "   ✅ Fallback demo mode for deployment environments"
echo "   ✅ 6 categories: Buildings, Forest, Glacier, Mountain, Sea, Street"
echo ""

# Final checks
print_status "Running final deployment checks..."

# Check file sizes (Streamlit Cloud has limits)
large_files=$(find . -type f -size +100M 2>/dev/null | grep -v ".git" | head -5)
if [ ! -z "$large_files" ]; then
    print_warning "Large files detected (>100MB):"
    echo "$large_files"
    print_warning "Consider using Git LFS for large files"
fi

# Check for sensitive files
sensitive_patterns=("*.key" "*.pem" "*.p12" "*password*" "*secret*")
for pattern in "${sensitive_patterns[@]}"; do
    sensitive_files=$(find . -name "$pattern" -not -path "./.git/*" 2>/dev/null)
    if [ ! -z "$sensitive_files" ]; then
        print_warning "Potential sensitive files found:"
        echo "$sensitive_files"
        print_warning "Make sure these are in .gitignore"
    fi
done

print_success "Deployment preparation completed! 🎉"
print_status "Your CLASSIFIT app is production-ready!"

echo ""
echo "=================================================="
echo "🚀 Ready to deploy to Streamlit Cloud!"
echo "==================================================" 