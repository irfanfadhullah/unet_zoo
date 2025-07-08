import os
import datetime

class Logger:
    """Enhanced logging utility for both console and file output"""
    
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_file = open(log_file_path, 'w', encoding='utf-8')
        
        self.log_both("="*80)
        self.log_both(f"BONE SEGMENTATION EXPERIMENT LOG")
        self.log_both(f"Experiment started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_both("="*80)
        
    def log_both(self, message: str):
        """Log message to both console and file"""
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
        
    def log_file_only(self, message: str):
        """Log message to file only"""
        self.log_file.write(message + '\n')
        self.log_file.flush()
        
    def close(self):
        """Close the log file"""
        self.log_both("="*80)
        self.log_both(f"Experiment completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_both("="*80)
        self.log_file.close()