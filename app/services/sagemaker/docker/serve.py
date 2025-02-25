#!/usr/bin/env python

from sagemaker_inference import model_server

HANDLER_SERVICE = "inference:handler_service"

if __name__ == "__main__":
    model_server.start_model_server(handler_service=HANDLER_SERVICE)