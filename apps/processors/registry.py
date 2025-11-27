# apps/processors/registry.py
"""
Operation Registry for media processing operations.

This module provides a centralized registry for all processing operations.
Operations are registered using a decorator and can be retrieved by name.
The registry also handles parameter validation against defined schemas.
"""

import logging
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import (
    OperationNotFoundError,
    OperationRegistrationError,
    InvalidParametersError,
)

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Supported media types for operations."""
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"


class ParameterType(Enum):
    """Supported parameter types for operation parameters."""
    INTEGER = "integer"
    STRING = "string"
    BOOLEAN = "boolean"
    FLOAT = "float"
    CHOICE = "choice"


@dataclass
class ParameterSchema:
    """
    Schema definition for an operation parameter.
    
    Attributes:
        param_name: The parameter name
        param_type: The type of the parameter
        required: Whether the parameter is required
        default: Default value if not provided
        description: Human-readable description
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        choices: List of valid choices for choice type
    """
    param_name: str
    param_type: ParameterType
    required: bool = False
    default: Any = None
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "param_name": self.param_name,
            "type": self.param_type.value,
            "required": self.required,
            "description": self.description,
        }
        if self.default is not None:
            result["default"] = self.default
        if self.min_value is not None:
            result["min"] = self.min_value
        if self.max_value is not None:
            result["max"] = self.max_value
        if self.choices is not None:
            result["choices"] = self.choices
        return result


@dataclass
class OperationDefinition:
    """
    Complete definition of a processing operation.
    
    Attributes:
        operation_name: Unique identifier for the operation
        media_type: Type of media this operation processes
        handler: The function that performs the operation
        parameters: List of parameter schemas
        description: Human-readable description
        input_formats: List of supported input formats
        output_formats: List of possible output formats
    """
    operation_name: str
    media_type: MediaType
    handler: Callable
    parameters: List[ParameterSchema] = field(default_factory=list)
    description: str = ""
    input_formats: List[str] = field(default_factory=list)
    output_formats: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for API responses."""
        return {
            "operation_name": self.operation_name,
            "media_type": self.media_type.value,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "input_formats": self.input_formats,
            "output_formats": self.output_formats,
        }


class OperationRegistry:
    """
    Singleton registry for all processing operations.
    
    This class maintains a central registry of all available operations
    and provides methods for registration, retrieval, and parameter validation.
    """
    
    _instance: Optional["OperationRegistry"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "OperationRegistry":
        """Create new singleton instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if not self._initialized:
            self._operations: Dict[str, OperationDefinition] = {}
            self._initialized = True
            logger.info("OperationRegistry initialized")
    
    def register_operation(
        self,
        operation_name: str,
        media_type: MediaType,
        handler: Callable,
        parameters: Optional[List[ParameterSchema]] = None,
        description: str = "",
        input_formats: Optional[List[str]] = None,
        output_formats: Optional[List[str]] = None,
    ) -> None:
        """
        Register an operation with the registry.
        
        Args:
            name: Unique identifier for the operation
            media_type: Type of media this operation processes
            handler: The function that performs the operation
            parameters: List of parameter schemas
            description: Human-readable description
            input_formats: List of supported input formats
            output_formats: List of possible output formats
            
        Raises:
            OperationRegistrationError: If registration fails
        """
        # Validate name
        if not operation_name or not isinstance(operation_name, str):
            raise OperationRegistrationError(operation_name or "unknown", "Name must be a non-empty string")
        
        if not operation_name.replace("_", "").isalnum():
            raise OperationRegistrationError(
                operation_name, 
                "Name must contain only alphanumeric characters and underscores"
            )
        
        # Check for duplicates
        if operation_name in self._operations:
            raise OperationRegistrationError(
                operation_name,
                f"Operation '{operation_name}' is already registered"
            )
        
        # Validate media type
        if not isinstance(media_type, MediaType):
            raise OperationRegistrationError(
                operation_name,
                f"Invalid media_type: {media_type}. Must be a MediaType enum value."
            )
        
        # Validate handler
        if not callable(handler):
            raise OperationRegistrationError(operation_name, "Handler must be callable function")
        
        # Create operation definition
        operation = OperationDefinition(
            operation_name=operation_name,
            media_type=media_type,
            handler=handler,
            parameters=parameters or [],
            description=description,
            input_formats=input_formats or [],
            output_formats=output_formats or [],
        )
        
        self._operations[operation_name] = operation
        logger.info(f"Registered operation: {operation_name} (media_type={media_type.value})")
    
    def get_operation(self, operation_name: str) -> OperationDefinition:
        """
        Retrieve an operation by name.
        
        Args:
            operation_name: The operation name to look up
            
        Returns:
            The OperationDefinition for the operation
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
        """
        if operation_name not in self._operations:
            raise OperationNotFoundError(operation_name)
        return self._operations[operation_name]
    
    def is_registered(self, operation_name: str) -> bool:
        """
        Check if an operation is registered.
        
        Args:
            operation_name: The operation name to check
            
        Returns:
            True if the operation exists, False otherwise
        """
        return operation_name in self._operations
    
    def list_registered_operations(self) -> List[OperationDefinition]:
        """
        Get all registered operations.
        
        Returns:
            List of all OperationDefinition objects
        """
        return list(self._operations.values())
    
    def list_operations_by_media_type(self, media_type: MediaType) -> List[OperationDefinition]:
        """
        Get all operations for a specific media type.
        
        Args:
            media_type: The media type to filter by
            
        Returns:
            List of OperationDefinition objects matching the media type
        """
        return [
            op for op in self._operations.values()
            if op.media_type == media_type
        ]
    
    def validate_parameters(
        self,
        operation_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate and normalize parameters for an operation.
        
        This method validates user-provided parameters against the operation's
        schema, applies defaults for missing optional parameters, and validates
        types and constraints.
        
        Args:
            operation_name: Name of the operation
            parameters: User-provided parameters
            
        Returns:
            Validated parameters with defaults applied
            
        Raises:
            OperationNotFoundError: If operation doesn't exist
            InvalidParametersError: If parameters are invalid
        """
        operation = self.get_operation(operation_name)
        validated_params: Dict[str, Any] = {}
        errors: List[str] = []
        
        # Create a set of parameter names for quick lookup
        provided_params = set(parameters.keys()) if parameters else set()
        defined_params = {p.name: p for p in operation.parameters}
        
        # Check for unknown parameters
        unknown_params = provided_params - set(defined_params.keys())
        if unknown_params:
            errors.append(f"Unknown parameters: {', '.join(unknown_params)}")
        
        # Validate each defined parameter
        for param_schema in operation.parameters:
            param_name = param_schema.name
            
            # Check if parameter is provided
            if param_name in parameters:
                value = parameters[param_name]
                
                # Validate and convert value
                try:
                    validated_value = self._validate_parameter_value(
                        param_name, value, param_schema
                    )
                    validated_params[param_name] = validated_value
                except ValueError as e:
                    errors.append(str(e))
            
            elif param_schema.required:
                # Required parameter not provided
                errors.append(f"Required parameter '{param_name}' is missing")
            
            elif param_schema.default is not None:
                # Apply default value
                validated_params[param_name] = param_schema.default
        
        # Raise exception if there are errors
        if errors:
            raise InvalidParametersError(operation_name, errors)
        
        return validated_params
    
    def _validate_parameter_value(
        self,
        param_name: str,
        value: Any,
        schema: ParameterSchema
    ) -> Any:
        """
        Validate and convert a single parameter value.
        
        Args:
            param_name: Name of the parameter
            value: Value to validate
            schema: Parameter schema
            
        Returns:
            Validated and converted value
            
        Raises:
            ValueError: If validation fails
        """
        # Type validation and conversion
        if schema.param_type == ParameterType.INTEGER:
            return self._validate_integer(param_name, value, schema)
        
        elif schema.param_type == ParameterType.FLOAT:
            return self._validate_float(param_name, value, schema)
        
        elif schema.param_type == ParameterType.BOOLEAN:
            return self._validate_boolean(param_name, value)
        
        elif schema.param_type == ParameterType.STRING:
            return self._validate_string(param_name, value)
        
        elif schema.param_type == ParameterType.CHOICE:
            return self._validate_choice(param_name, value, schema)
        
        else:
            raise ValueError(f"Unknown parameter type for '{param_name}'")
    
    def _validate_integer(
        self,
        param_name: str,
        value: Any,
        schema: ParameterSchema
    ) -> int:
        """Validate an integer parameter."""
        # Convert to integer
        try:
            int_value = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Parameter '{param_name}' must be an integer, got {type(value).__name__}"
            )
        
        # Check range
        if schema.min_value is not None and int_value < schema.min_value:
            raise ValueError(
                f"Parameter '{param_name}' must be at least {schema.min_value}, got {int_value}"
            )
        
        if schema.max_value is not None and int_value > schema.max_value:
            raise ValueError(
                f"Parameter '{param_name}' must be at most {schema.max_value}, got {int_value}"
            )
        
        return int_value
    
    def _validate_float(
        self,
        param_name: str,
        value: Any,
        schema: ParameterSchema
    ) -> float:
        """Validate a float parameter."""
        # Convert to float
        try:
            float_value = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"Parameter '{param_name}' must be a number, got {type(value).__name__}"
            )
        
        # Check range
        if schema.min_value is not None and float_value < schema.min_value:
            raise ValueError(
                f"Parameter '{param_name}' must be at least {schema.min_value}, got {float_value}"
            )
        
        if schema.max_value is not None and float_value > schema.max_value:
            raise ValueError(
                f"Parameter '{param_name}' must be at most {schema.max_value}, got {float_value}"
            )
        
        return float_value
    
    def _validate_boolean(self, param_name: str, value: Any) -> bool:
        """Validate a boolean parameter."""
        if isinstance(value, bool):
            return value
        
        # Handle string representations
        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in ("true", "1", "yes", "on"):
                return True
            if lower_value in ("false", "0", "no", "off"):
                return False
        
        # Handle numeric representations
        if isinstance(value, (int, float)):
            return bool(value)
        
        raise ValueError(
            f"Parameter '{param_name}' must be a boolean, got {type(value).__name__}"
        )
    
    def _validate_string(self, param_name: str, value: Any) -> str:
        """Validate a string parameter."""
        if value is None:
            raise ValueError(f"Parameter '{param_name}' cannot be None")
        return str(value)
    
    def _validate_choice(
        self,
        param_name: str,
        value: Any,
        schema: ParameterSchema
    ) -> Any:
        """Validate a choice parameter."""
        if schema.choices is None or len(schema.choices) == 0:
            raise ValueError(f"Parameter '{param_name}' has no defined choices")
        
        # Convert value to string for comparison if choices are strings
        str_value = str(value)
        str_choices = [str(c) for c in schema.choices]
        
        if str_value in str_choices:
            # Return the original choice value (preserving type)
            idx = str_choices.index(str_value)
            return schema.choices[idx]
        
        raise ValueError(
            f"Parameter '{param_name}' must be one of {schema.choices}, got '{value}'"
        )
    
    def get_operation_info(self, name: str) -> Dict[str, Any]:
        """
        Get operation information as a dictionary.
        
        Args:
            name: The operation name
            
        Returns:
            Dictionary with operation details
        """
        operation = self.get_operation(name)
        return operation.to_dict()
    
    def clear(self) -> None:
        """
        Clear all registered operations.
        
        This is primarily useful for testing.
        """
        self._operations.clear()
        logger.info("OperationRegistry cleared")


# Global registry instance
registry = OperationRegistry()


def register_operation(
    operation_name: str,
    media_type: MediaType,
    parameters: Optional[List[ParameterSchema]] = None,
    description: str = "",
    input_formats: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator for registering a processing operation.
    
    Args:
        operation_name: Unique identifier for the operation
        media_type: Type of media this operation processes
        parameters: List of parameter schemas
        description: Human-readable description
        input_formats: List of supported input formats
        output_formats: List of possible output formats
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        registry.register_operation(
            name=operation_name,
            media_type=media_type,
            handler=func,
            parameters=parameters,
            description=description,
            input_formats=input_formats,
            output_formats=output_formats,
        )
        return func
    return decorator


def get_registry() -> OperationRegistry:
    """
    Get the global operation registry instance.
    
    Returns:
        The singleton OperationRegistry instance
    """
    return registry