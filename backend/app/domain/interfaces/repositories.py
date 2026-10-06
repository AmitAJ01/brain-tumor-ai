"""
Repository Pattern Interfaces for Dataset and Model Persistence
Decouples domain/application logic from direct filesystem and database I/O.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

class IDatasetRepository(ABC):
    """
    Repository interface for brain tumor MRI dataset access.
    Abstracts dataset reading, format conversion (.mat / .png),
    and partition loading (train, val, test).
    """

    @abstractmethod
    def get_summary(self) -> Dict[str, Any]:
        """Return dataset statistics, class distribution, and split counts."""
        pass

    @abstractmethod
    def get_sample_by_id(self, sample_id: str) -> Dict[str, Any]:
        """Fetch image, label, patient ID, and mask for a specific slice."""
        pass

    @abstractmethod
    def list_samples(self, split: Optional[str] = None) -> List[str]:
        """List sample identifiers for an optional split (train, val, test)."""
        pass

class IModelRepository(ABC):
    """
    Repository interface for model artifact storage and metadata management.
    """

    @abstractmethod
    def save_checkpoint(self, model_name: str, state_dict: dict, metrics: Dict[str, Any]) -> str:
        """Persist model weights and associated evaluation metadata to disk."""
        pass

    @abstractmethod
    def load_checkpoint(self, model_name: str) -> Dict[str, Any]:
        """Load model state dict and associated metadata from disk."""
        pass

    @abstractmethod
    def exists(self, model_name: str) -> bool:
        """Check if trained model artifact exists."""
        pass

    @abstractmethod
    def get_path(self, model_name: str) -> Path:
        """Get filesystem path for model artifact."""
        pass
