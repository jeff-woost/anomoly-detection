#!/bin/bash
# Daily execution script for Trade Anomaly Detection System

# Load environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export ANOMALY_DETECTOR_HOME="$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Log file
LOG_FILE="logs/daily_run_$(date +%Y%m%d).log"

echo "Starting daily anomaly detection run at $(date)" >> "$LOG_FILE"

# Backup database
echo "Backing up database..." >> "$LOG_FILE"
cp trade_anomalies.db "backups/trade_anomalies_$(date +%Y%m%d_%H%M%S).db"

# Run anomaly detection
echo "Running anomaly detection..." >> "$LOG_FILE"
python3 scheduler.py --run-once >> "$LOG_FILE" 2>&1

# Generate daily report
echo "Generating daily report..." >> "$LOG_FILE"
python3 generate_report.py --type daily --date today >> "$LOG_FILE" 2>&1

# Clean up old logs (keep last 30 days)
find logs/ -name "*.log" -mtime +30 -delete

echo "Daily run completed at $(date)" >> "$LOG_FILE"

# Send notification (optional)
# python3 send_notification.py --message "Daily anomaly detection completed"