#!/usr/bin/env python3
"""
Model Persistence System
========================

Sistema per salvare e caricare i modelli allenati per evitare re-training.
"""

import os
import pickle
import json
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class ModelStorage:
    """Gestione persistenza modelli"""
    
    def __init__(self, storage_dir: str = "models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    def model_exists(self, league_code: str) -> bool:
        """Controlla se un modello esiste già su disco"""
        model_file = self.storage_dir / f"{league_code}_model.pkl"
        return model_file.exists()
        
    def is_model_fresh(self, league_code: str, max_age_hours: int = 24) -> bool:
        """Controlla se il modello è ancora fresco (non troppo vecchio)"""
        if not self.model_exists(league_code):
            return False
            
        metadata = self.get_model_metadata(league_code)
        if not metadata:
            return False
            
        try:
            saved_at = datetime.fromisoformat(metadata['saved_at'])
            age_hours = (datetime.now() - saved_at).total_seconds() / 3600
            return age_hours <= max_age_hours
        except:
            return False
        
    def save_model(self, model: Any, league_code: str, metadata: Dict = None) -> bool:
        """Salva un modello allenato su disco"""
        try:
            model_file = self.storage_dir / f"{league_code}_model.pkl"
            metadata_file = self.storage_dir / f"{league_code}_metadata.json"
            
            # Salva il modello
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Salva i metadati
            metadata = metadata or {}
            metadata.update({
                'league_code': league_code,
                'saved_at': datetime.now().isoformat(),
                'model_class': model.__class__.__name__,
                'training_data_size': getattr(model, 'training_data_size', 0),
                'is_trained': getattr(model, 'is_trained', False)
            })
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model {league_code} saved to {model_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model {league_code}: {e}")
            return False
    
    def load_model(self, league_code: str) -> Optional[Any]:
        """Carica un modello salvato da disco"""
        try:
            model_file = self.storage_dir / f"{league_code}_model.pkl"
            
            if not model_file.exists():
                print(f"📁 No saved model found for {league_code}")
                return None
            
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            print(f"✅ Model {league_code} loaded from {model_file}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading model {league_code}: {e}")
            return None
    
    def get_model_metadata(self, league_code: str) -> Optional[Dict]:
        """Ottieni metadati di un modello salvato"""
        try:
            metadata_file = self.storage_dir / f"{league_code}_metadata.json"
            
            if not metadata_file.exists():
                return None
            
            with open(metadata_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"❌ Error loading metadata {league_code}: {e}")
            return None
    
    def list_saved_models(self) -> Dict[str, Dict]:
        """Lista tutti i modelli salvati con metadati"""
        models = {}
        
        for model_file in self.storage_dir.glob("*_model.pkl"):
            league_code = model_file.stem.replace('_model', '')
            metadata = self.get_model_metadata(league_code)
            
            models[league_code] = {
                'file_path': str(model_file),
                'file_size': model_file.stat().st_size,
                'metadata': metadata
            }
        
        return models
    
    def delete_model(self, league_code: str) -> bool:
        """Elimina un modello salvato"""
        try:
            model_file = self.storage_dir / f"{league_code}_model.pkl"
            metadata_file = self.storage_dir / f"{league_code}_metadata.json"
            
            deleted = False
            if model_file.exists():
                model_file.unlink()
                deleted = True
            
            if metadata_file.exists():
                metadata_file.unlink()
                deleted = True
            
            if deleted:
                print(f"✅ Model {league_code} deleted")
                return True
            else:
                print(f"❌ Model {league_code} not found")
                return False
                
        except Exception as e:
            print(f"❌ Error deleting model {league_code}: {e}")
            return False

# Istanza globale
model_storage = ModelStorage()

def save_trained_model(model, league_code: str) -> bool:
    """Helper function per salvare modelli"""
    metadata = {
        'training_matches': getattr(model, 'training_data_size', 0),
        'accuracy_metrics': getattr(model, 'model_stats', {}),
        'model_version': 'EXACT_REPLICA'
    }
    return model_storage.save_model(model, league_code, metadata)

def load_trained_model(league_code: str):
    """Helper function per caricare modelli"""
    return model_storage.load_model(league_code)

def list_available_models():
    """Lista modelli disponibili"""
    return model_storage.list_saved_models()

if __name__ == "__main__":
    # Test del sistema
    print("🧪 Testing Model Storage System")
    print("="*50)
    
    # Lista modelli esistenti
    models = list_available_models()
    print(f"📋 Found {len(models)} saved models:")
    
    for league, info in models.items():
        metadata = info['metadata']
        if metadata:
            print(f"  ✅ {league}: trained={metadata.get('is_trained', False)}, "
                  f"size={info['file_size']} bytes, "
                  f"saved={metadata.get('saved_at', 'unknown')}")
        else:
            print(f"  ⚠️  {league}: no metadata")