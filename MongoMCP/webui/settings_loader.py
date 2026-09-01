"""Select the active settings module via the SETTINGS_MODULE env var.

Defaults to `aws_settings` so existing AWS/local deployments are unchanged.
Set SETTINGS_MODULE=kanopy_settings on Kanopy to source secrets from the K8s
secret store instead of AWS Secrets Manager.
"""
import importlib
import os

_MODULE_NAME = os.getenv("SETTINGS_MODULE", "aws_settings")
settings = importlib.import_module(_MODULE_NAME).settings
