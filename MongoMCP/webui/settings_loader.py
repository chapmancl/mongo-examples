"""Select the active settings module via the SETTINGS_MODULE env var.

Defaults to `aws_settings` so existing AWS/local deployments are unchanged.
Set SETTINGS_MODULE=kanopy_settings on Kanopy to source secrets from the K8s
secret store instead of AWS Secrets Manager.
"""
import importlib
import os

_MODULE_NAME = os.getenv("SETTINGS_MODULE", "aws_settings")
settings = importlib.import_module(_MODULE_NAME).settings

# shim to use the local settings when USE_LOCAL_MODE is set. 
# you can also just set local_settings in the SETTINGS_MODULE env var if desired.
USE_LOCAL_MODE = os.getenv('USE_LOCAL_MODE', 'false').lower() == "true"
if USE_LOCAL_MODE:
    settings = importlib.import_module("local_settings").settings

    


