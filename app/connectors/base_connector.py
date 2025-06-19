from abc import ABC, abstractmethod
from typing import List, Any, Optional, Dict
from config import get_settings
from models.document import Document

class BaseConnector(ABC):
    """Base class for all data source connectors"""
    
    def __init__(self):
        self.settings = get_settings()
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to the data source"""
        pass
    
    @abstractmethod
    async def fetch_data(self, **kwargs) -> List[Document]:
        """Fetch data from the source and return as Documents"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health status of the data source"""
        pass
    
    def get_connector_info(self) -> Dict[str, Any]:
        """Return connector metadata"""
        return {
            "name": self.__class__.__name__,
            "type": "base",
            "status": "unknown"
        }