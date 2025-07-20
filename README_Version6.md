# Trade Anomaly Detection System

An enterprise-grade, high-performance anomaly detection system for trading data and P&L analysis. This system combines Python machine learning capabilities with C++ performance optimization to provide real-time anomaly detection for trading operations.

## Features

- **Real-time Anomaly Detection**: Detects unusual trading patterns, extreme P&L movements, and risk limit breaches
- **Hybrid Architecture**: Python for ML models and GUI, C++ for high-performance statistical computations
- **Multiple Detection Methods**:
  - Isolation Forest for general anomaly detection
  - Autoencoder neural networks for complex pattern recognition
  - Statistical methods (Z-score, MAD, percentile analysis)
  - DBSCAN clustering for group anomalies
- **Professional Enterprise GUI**: PyQt5-based interface with dark theme
- **Automated Scheduling**: Daily automated analysis with configurable intervals
- **Comprehensive Logging**: Detailed logging for audit and debugging
- **Database Storage**: SQLite backend for anomaly history and model performance tracking

## System Requirements

### Operating System
- Linux (Ubuntu 20.04+ recommended)
- Windows 10/11 with WSL2
- macOS 10.15+

### Hardware Requirements
- CPU: 4+ cores recommended (8+ for optimal performance)
- RAM: 16GB minimum (32GB recommended)
- Storage: 10GB free space for data and models

### Software Prerequisites
- Python 3.8 or higher
- C++ compiler with C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.14 or higher
- Git

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/trade-anomaly-detector.git
cd trade-anomaly-detector