#!/bin/bash
# Setup script for Trade Anomaly Detection System

echo "Trade Anomaly Detection System Setup"
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8"

if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✓ Python version check passed: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "Error: No C++ compiler found. Please install g++ or clang++"
    exit 1
fi

echo "✓ C++ compiler found"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "Error: CMake not found. Please install CMake 3.14 or higher"
    exit 1
fi

cmake_version=$(cmake --version | head -n1 | awk '{print $3}')
echo "✓ CMake found: version $cmake_version"

# Build C++ extension
echo "Building C++ extension..."
mkdir -p cpp_src/build
cd cpp_src/build

if cmake .. && make -j$(nproc); then
    echo "✓ C++ extension built successfully"
else
    echo "Error: Failed to build C++ extension"
    exit 1
fi

cd ../..

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data logs reports models

# Copy configuration
if [ ! -f config.json ]; then
    cp config.example.json config.json
    echo "✓ Configuration file created (config.json)"
fi

# Initialize database
echo "Initializing database..."
python3 -c "
from trade_anomaly_detector import MLAnomalyDetector
detector = MLAnomalyDetector()
print('✓ Database initialized')
"

# Run tests
echo "Running tests..."
if pytest tests/ -v --tb=short; then
    echo "✓ All tests passed"
else
    echo "Warning: Some tests failed"
fi

echo ""
echo "Setup completed successfully!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the GUI, run:"
echo "  python trade_anomaly_gui.py"
echo ""
echo "To run anomaly detection from command line:"
echo "  python detect_anomalies.py --help"