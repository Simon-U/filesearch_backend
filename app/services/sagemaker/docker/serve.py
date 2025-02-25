from sagemaker_inference import model_server

# Use the module and variable name. Here we assume your file is named "inference.py".
HANDLER_SERVICE = "inference:default_handler"

if __name__ == '__main__':
    model_server.start_model_server(handler_service=HANDLER_SERVICE)