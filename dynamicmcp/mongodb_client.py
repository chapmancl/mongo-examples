"""
MongoDB Client connection management using settings from settings_aws.py
"""

import logging
from pymongo.errors import PyMongoError
from motor.motor_asyncio import AsyncIOMotorClient
import pymongo

# Import settings
from settings_aws import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client = None
        self.db = None
        self.collection = None
        self._connection_initialized = False
        self._db_name = None
        self._collection_name = None
    
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
            self.client = AsyncIOMotorClient(settings.get_mongo_uri())
            
            # Test the connection
            ping_result = await self.client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {self._db_name}")
            
            self.db = self.client[self._db_name]
            self.collection = self.db[self._collection_name]
            self._connection_initialized = True
            
        except PyMongoError as e:
            ip_address = self.get_current_ip()
            logger.error(f"Failed to connect to MongoDB from ip: {ip_address}: {e}")
            self._connection_initialized = False            
        return ping_result

    def sync_connect_to_mongodb(self):
        """Synchronous version of connect_to_mongodb"""
        self.client =  pymongo.MongoClient(settings.get_mongo_uri())
        self.client.admin.command('ping')
        self.db = self.client[self._db_name]    
        self.collection = self.db[self._collection_name]
        self._connection_initialized = True

