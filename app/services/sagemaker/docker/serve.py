from sagemaker_inference import model_server

if __name__ == '__main__':
    # This tells the server to look for your handler functions in the module specified
    # by the SAGEMAKER_PROGRAM environment variable (in your case, inference.py)
    model_server.start_model_server(handler_service='inference:default_handler')