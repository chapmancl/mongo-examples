import time
import traceback
import logging
from requests.auth import HTTPDigestAuth
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings =  {
        "org_id":"",
        "project_id":"",
        "public_key":"",
        "private_key":"",
        "db_name": "sample_airbnb",
        "collection_name": "listingsAndReviews",
        "stream_instance": "vectorDemo",
        "cluster_connection_name": "airbnb-data",
        "endpoint_connection_name": "mcp-cluster-vectorize",
        "processor_name": "AirBnB-ASP-Processor"        
    }
# Headers
HEADERS = {
    'Accept': 'application/vnd.atlas.2024-05-30+json',
    'Content-Type': 'application/json'
}


PIPELINE = [
    {
       "$source": {  
            "connectionName": settings["connection_name"],
            "db": settings["db_name"],
            "coll": settings["collection_name"],
        "config": {
            "fullDocument": "required",
            "pipeline": [
                {  
                "$match": {  
                    "operationType": { "$in": ["insert", "update"] },
                    
                }}
            ]
        }}                             
    },
    {  
    "$addFields": {  
        "fullDocument.concatenated_text": {  
        "$trim": {  
            "input": {  
            "$concat": [  
                { "$ifNull": ["$fullDocument.name", ""] }, " ",  
                { "$ifNull": ["$fullDocument.summary", ""] }, " ",  
                { "$ifNull": ["$fullDocument.space", ""] }, " ",  
                { "$ifNull": ["$fullDocument.description", ""] }, " ",  
                { "$ifNull": ["$fullDocument.property_type", ""] }, " ",  
                { "$ifNull": ["$fullDocument.room_type", ""] }, " ",  
                { "$ifNull": ["$fullDocument.bed_type", ""] } 
            ]  
            }  
        }  
        }
    }},
    {
    "$https": {
        "connectionName": settings["endpoint_connection_name"],
        "method": "POST",
        "as": "apiResults",
        "config": {"parseJsonStrings": True},
        "payload": [ {"$project": {
                "_id": 1,
                "textChunk": "$fullDocument.concatenated_text"
                }}
        ]
    }},

{  
  "$set": {  
    "fullDocument.embedding": {  
      "$cond": {  
        "if": { "$not": { "$ifNull": ["$apiResults.error", False] } },  
        "then": "$apiResults.vector",  
        "else": "$$REMOVE"
      }  
    },  
    "fullDocument.embedding_status": {  
      "$cond": {  
        "if": { "$ifNull": ["$apiResults.error", False] },  
        "then": "failed",  
        "else": "success"  
      }  
    },  
    "fullDocument.embedding_error": {  
      "$cond": {  
        "if": { "$ifNull": ["$apiResults.error", False] },  
        "then": "$apiResults.error",  
        "else": "$$REMOVE"  
      }  
    }
  }  
},
{
    "$unset": ["fullDocument.concatenated_text"]
},
{
    "$replaceRoot": {
        "newRoot": "$fullDocument"
    }
},
    
    {
    "$merge": {
        "into": {
        "connectionName": settings["cluster_connection_name"],
        "db": settings["db_name"],
        "coll": settings["collection_name"]
        },        
        "whenMatched": "merge"
    }}
]

class MongoDBASP():
    def __init__(self):
        super().__init__()
        self._db_name = settings["db_name"]
        self._collection_name = settings["collection_name"]
        self.base_url = "https://cloud.mongodb.com/api/atlas/v2"
        self.stream_instance = settings["stream_instance"]
        self.org_id: str = settings["org_id"]
        self.public_key: str = settings["public_key"]
        self.private_key: str = settings["private_key"]
        self.project_id: str = settings["project_id"]
    
    def _get_auth(self) -> HTTPDigestAuth:
        """Create authentication object for API requests."""
        return HTTPDigestAuth(self.public_key, self.private_key)

    def run(self):
        try:
            logger.info("Starting AirBnB ASP...")
            #result = self.create_stream_processor(PIPELINE)
            result = self.update_stream_processor(PIPELINE)

            logger.info("AirBnB ASP completed successfully.")
            return result
        except Exception as e:
            logger.error(f"Error running AirBnB ASP: {e}")
            traceback.print_exc()
            return None

    def update_stream_processor(self, pipeline):  
        """  
        Create a new stream processor with the given pipeline  
        """  
        url = f"{self.base_url}/groups/{self.project_id}/streams/{self.stream_instance}/processor/{settings["processor_name"]}"  
        
        # stop it first
        stopresponse = requests.post(  
            f"{url}:stop",                           
            headers=HEADERS,
            auth=self._get_auth()
        )  
        print(f"Stopped processor response: {stopresponse.status_code}")

        # Stream processor configuration  
        processor_config = {  
            "name": settings["processor_name"],  
            "pipeline": pipeline,
            "options": {  
                "dlq": {  
                    "connectionName": settings["cluster_connection_name"],
                    "coll": "vector_stream_dlq",  
                    "db": settings["db_name"]
                }  
            }  
        }  
          
        response = requests.patch(  
            url,  
            json=processor_config,              
            headers=HEADERS,
            auth=self._get_auth()
        )  
          
        if response.status_code == 200:  
            print(f"Stream processor '{settings["processor_name"]}' updated successfully!")  

            startresponse = requests.post(  
                f"{url}:start",                           
                headers=HEADERS,
                auth=self._get_auth()
            )  
            print(f"Start processor response: {startresponse.status_code}")

            return response.json()  
        else:  
            print(f"Error creating stream processor: {response.status_code}")  
            print(response.text)  
            return None  
        

    def create_stream_processor(self, pipeline):  
        """  
        Create a new stream processor with the given pipeline  
        """  
        url = f"{self.base_url}/groups/{self.project_id}/streams/{self.stream_instance}/processor"  
          
        # Stream processor configuration  
        processor_config = {  
            "name": settings["processor_name"],  
            "pipeline": pipeline,
            "options": {  
                "dlq": {  
                    "connectionName": settings["cluster_connection_name"],
                    "coll": "stream_dlq",  
                    "db": settings["db_name"] 
                }  
            }  
        }  
          
        response = requests.post(  
            url,  
            json=processor_config,              
            headers=HEADERS,
            auth=self._get_auth()
        )  
          
        if response.status_code == 200:  
            print(f"Stream processor '{settings["processor_name"]}' created successfully!")  
            time.sleep(2)  # wait for 2 seconds before starting
            startresponse = requests.post(  
                f"{url}:start",                           
                headers=HEADERS,
                auth=self._get_auth()
            )  
            print(f"Start processor response: {startresponse.status_code}")

            return response.json()  
        else:  
            print(f"Error creating stream processor: {response.status_code}")  
            print(response.text)  
            return None  
  


def main():    
    asp = MongoDBASP()
    res = asp.run()
    #print(res)

if __name__ == "__main__":
    main()
