"""Vercel serverless entry point — re-exports the Flask app."""
import sys
import os

# Add parent directory to path so we can import stamp_webapp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stamp_webapp import app
