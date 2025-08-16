# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading and initial preprocessing of SMS spam dataset
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load SMS spam dataset from CSV file
        
        Expected format:
        - Column 1: 'label' (spam/ham)
        - Column 2: 'message' (SMS text)
        """
        try:
            # Try different encodings in case of encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.file_path, encoding=encoding)
                    logger.info(f"Data loaded successfully with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
                    
            if self.data is None:
                raise ValueError("Could not load data with any encoding")
                
            # Ensure correct column names
            if len(self.data.columns) >= 2:
                self.data.columns = ['label', 'message'] + list(self.data.columns[2:])
            
            # Basic data validation
            self._validate_data()
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self):
        """Validate the loaded data"""
        if self.data is None:
            raise ValueError("No data loaded")
            
        if 'label' not in self.data.columns or 'message' not in self.data.columns:
            raise ValueError("Required columns 'label' and 'message' not found")
            
        # Remove duplicates
        initial_shape = self.data.shape
        self.data = self.data.drop_duplicates()
        logger.info(f"Removed {initial_shape[0] - self.data.shape[0]} duplicate rows")
        
        # Remove null values
        self.data = self.data.dropna(subset=['label', 'message'])
        
        # Standardize labels
        self.data['label'] = self.data['label'].str.lower()
        unique_labels = self.data['label'].unique()
        logger.info(f"Dataset loaded: {self.data.shape[0]} samples, Labels: {unique_labels}")
        
    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """Split data into training and testing sets"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        X = self.data['message']
        y = self.data['label']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_data_info(self) -> dict:
        """Get basic information about the dataset"""
        if self.data is None:
            return {}
            
        info = {
            'total_samples': len(self.data),
            'spam_count': len(self.data[self.data['label'] == 'spam']),
            'ham_count': len(self.data[self.data['label'] == 'ham']),
            'spam_percentage': (len(self.data[self.data['label'] == 'spam']) / len(self.data)) * 100
        }
        return info