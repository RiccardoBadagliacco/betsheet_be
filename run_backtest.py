#!/usr/bin/env python3
"""
Script di utilità per eseguire il backtest dalla directory root del progetto
"""

import os
import sys

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Change to backtest directory and run
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest'))

# Import and run backtest
from football_backtest_real import main

if __name__ == "__main__":
    print("🧪 RUNNING FOOTBALL BACKTEST FROM ROOT")
    print("=" * 50)
    main()