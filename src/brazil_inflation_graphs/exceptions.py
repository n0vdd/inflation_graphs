"""Custom exceptions for Brazil inflation data fetcher."""


class BrazilInflationError(Exception):
    """Base exception for Brazil inflation data operations."""
    
    def __init__(self, message: str, source: str = None, details: dict = None):
        super().__init__(message)
        self.source = source
        self.details = details or {}


class DataFetchError(BrazilInflationError):
    """Exception raised when data fetching fails."""
    pass


class APIError(DataFetchError):
    """Exception raised for API-related errors."""
    
    def __init__(self, message: str, status_code: int = None, source: str = None, details: dict = None):
        super().__init__(message, source, details)
        self.status_code = status_code


class DataValidationError(BrazilInflationError):
    """Exception raised when data validation fails."""
    pass


class DataProcessingError(BrazilInflationError):
    """Exception raised during data processing operations."""
    pass


class ConfigurationError(BrazilInflationError):
    """Exception raised for configuration-related errors."""
    pass


class StorageError(BrazilInflationError):
    """Exception raised for data storage operations."""
    pass


class FREDAPIError(APIError):
    """Specific exception for FRED API errors."""
    pass


class WorldBankAPIError(APIError):
    """Specific exception for World Bank API errors."""
    pass


class IBGESIDRAError(APIError):
    """Specific exception for IBGE SIDRA API errors."""
    pass