class FileLoaderError(Exception):
    """Base exception for file loader errors"""
    pass


class StorageNotConfiguredError(FileLoaderError):
    """Raised when trying to use a storage type that wasn't configured"""
    pass


class UnsupportedFileTypeError(FileLoaderError):
    """Raised when trying to load an unsupported file type"""
    pass