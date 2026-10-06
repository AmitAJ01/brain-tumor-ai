"""
Model Factory Implementation
Implements Factory Pattern for instantiating model adapters without exposing
concrete classes, constructors, or internal hyperparameters to the application layer.
"""

from typing import Optional, Dict, Type
from backend.app.domain.interfaces.model_adapter import ModelAdapter
from backend.app.infrastructure.adapters.efficientnet_adapter import EfficientNetAdapter
from backend.app.infrastructure.adapters.efficientnet_unet_adapter import EfficientNetUNetAdapter

class ModelFactory:
    """
    Factory Pattern for ML Model Adapters.
    Centralizes creation logic, hardware device binding, and model type resolution.
    """

    _ADAPTER_REGISTRY: Dict[str, Type[ModelAdapter]] = {
        "efficientnet": EfficientNetAdapter,
        "classifier": EfficientNetAdapter,
        "efficientnet_b0": EfficientNetAdapter,
        "efficient_unet": EfficientNetUNetAdapter,
        "segmentation": EfficientNetUNetAdapter,
        "efficientnet_unet": EfficientNetUNetAdapter,
    }

    @classmethod
    def create(cls, model_type: str, weights_path: Optional[str] = None, device: str = "cpu", **kwargs) -> ModelAdapter:
        """
        Create and optionally load a model adapter by type name.
        
        Args:
            model_type: Key identifying model ("efficientnet", "efficient_unet", etc.)
            weights_path: Optional path to checkpoint file to immediately load
            device: Computing device ('cpu', 'cuda')
            kwargs: Extra parameters passed to the adapter constructor
            
        Returns:
            Instance of ModelAdapter
        """
        normalized_type = model_type.lower().strip()
        adapter_cls = cls._ADAPTER_REGISTRY.get(normalized_type)

        if adapter_cls is None:
            available = list(cls._ADAPTER_REGISTRY.keys())
            raise ValueError(
                f"Unknown model type '{model_type}'. Available options: {available}"
            )

        adapter = adapter_cls(device=device, **kwargs)

        if weights_path is not None:
            adapter.load_model(weights_path=weights_path, device=device)

        return adapter

    @classmethod
    def register(cls, model_type: str, adapter_cls: Type[ModelAdapter]) -> None:
        """Register a new model adapter type (Open/Closed Principle)."""
        cls._ADAPTER_REGISTRY[model_type.lower().strip()] = adapter_cls
