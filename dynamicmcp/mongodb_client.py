"""
MongoDB Client connection management using settings from settings_aws.py
"""

import logging
from typing import Any, Dict, List, Tuple
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo
import requests

# Import settings
from settings_aws import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db_url = None # set this if we're going to a cluster that is not our default from settings
        self.db = None
        self.collection = None
        self._connection_initialized = False
        # default should always be the config collection because it is the only one we know about at first
        self._db_name = settings.mcp_config_db
        self._collection_name = settings.mcp_config_col

    def set_config(self, config: Dict) -> None:
        """Override the default tool configuration from a dictionary"""
        if config is None:
            raise ValueError("Config cannot be None. Check env variables and AWS secrets.")
        # override the mongo url from our settings
        self.db_url =           config["url"]
        self._db_name =         config['database']
        self._collection_name = config['collection']
   
    def get_mongo_uri(self) -> str:
        """
        Get the complete MongoDB connection URI.
        
        Returns:
            MongoDB connection string
        """
        credentials = settings.get_mongo_credentials()
        # the url may be overidden by an incoming dynamic config, so test that here and return the local one instead of the settings based url
        m_url = settings.mongo_url()
        if self.db_url:
            m_url = self.db_url
        return f"mongodb+srv://{credentials['username']}:{credentials['password']}@{m_url}"

    def get_current_ip(self) -> str:
        """
        Get the current public IP address using AWS's checkip service.
        useful for logging network issues
        """
        try:
            # Make request to AWS checkip service with timeout
            response = requests.get('https://checkip.amazonaws.com', timeout=10)
            response.raise_for_status()  # Raise exception for bad status codes
            
            ip_address = response.text.strip()
            
            return ip_address
        except Exception as e:
            logger.error(f"Error fetching current IP: {e}")
            return f"Error fetching current IP: {e}"

    async def ensure_connection(self):
        """Ensure MongoDB connection is established"""
        print(f"connecting to mongodb {self._db_name} {self._collection_name}")
        if not self._connection_initialized:
            return await self.connect_to_mongodb()        
        return await self.client.admin.command('ping')     
    
    async def connect_to_mongodb(self):
        """Initialize MongoDB connection using settings.py configuration"""
        ping_result = None
        try:
            self.client = AsyncIOMotorClient(self.get_mongo_uri())
            
            # Test the connection
            ping_result = await self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {self._db_name}")
            
            self._set_locals()
            self._connection_initialized = True
            # load all tools to return configs
            self.ALLTOOLS = await self.collection.distinct("Name",{ "active": True})
            
        except PyMongoError as e:
            ip_address = self.get_current_ip()
            logger.error(f"Failed to connect to MongoDB from ip: {ip_address}: {e}")
            self._connection_initialized = False            
        return ping_result

    def sync_connect_to_mongodb(self):
        """Synchronous version of connect_to_mongodb"""
        try:
            self.client =  pymongo.MongoClient(self.get_mongo_uri())
            self.client.admin.command('ping')
            self._set_locals()
            self._connection_initialized = True
        except Exception as e:
            ip_address = self.get_current_ip()
            self._connection_initialized = False
            raise ConnectionError(f"Failed to connect to MongoDB from ip: {ip_address}: \r\n{e}")
        return self._connection_initialized

    def _set_locals(self):
        """Set local database and collection references if we have the settings"""
        if self._db_name:
            self.db = self.client[self._db_name]    
        if self._collection_name:
            self.collection = self.db[self._collection_name]

