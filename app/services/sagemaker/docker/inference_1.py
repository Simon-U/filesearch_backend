import json
import logging

# Set up basic logging to stdout
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

def model_fn(model_dir):
    """
    Load the model.
    This is called by SageMaker when starting the model server.
    
    Args:
        model_dir: Directory where model artifacts are stored
        
    Returns:
        A simple model representation (in this case, just a string)
    """
    logger.info(f"model_fn called with model_dir: {model_dir}")
    # For this simple example, our "model" is just a string
    return "Echo Model"

def input_fn(request_body, request_content_type):
    """
    Transform input data.
    
    Args:
        request_body: The request payload
        request_content_type: The content type of the request
        
    Returns:
        The input data
    """
    logger.info(f"input_fn called with content_type: {request_content_type}")
    logger.info(f"request_body: {request_body}")
    
    # For JSON inputs
    if request_content_type == 'application/json':
        try:
            input_data = json.loads(request_body)
            logger.info(f"Parsed JSON input: {input_data}")
            return input_data
        except Exception as e:
            logger.error(f"Error parsing JSON: {e}")
            raise
    else:
        # For any other content type, just return as is
        logger.info(f"Non-JSON input received with content type: {request_content_type}")
        return request_body

def predict_fn(input_data, model):
    """
    Generate prediction.
    
    Args:
        input_data: Preprocessed input data from input_fn
        model: The model loaded in model_fn
        
    Returns:
        The prediction result (in this case, just echo back the input)
    """
    logger.info(f"predict_fn called with model: {model}")
    logger.info(f"input_data: {input_data}")
    
    # Simply echo back the input data and add a message
    return {
        "received_input": input_data,
        "message": "Hello from SageMaker Echo Model!",
        "model_used": model
    }

def output_fn(prediction, response_content_type):
    """
    Transform prediction output.
    
    Args:
        prediction: Model prediction from predict_fn
        response_content_type: Expected content type of the response
        
    Returns:
        Formatted response
    """
    logger.info(f"output_fn called with response_content_type: {response_content_type}")
    logger.info(f"prediction: {prediction}")
    
    # Default to JSON if not specified
    if response_content_type is None or response_content_type == 'application/json':
        return json.dumps(prediction), 'application/json'
    else:
        # For this example, we only support JSON
        logger.warning(f"Unsupported content type: {response_content_type}, using JSON instead")
        return json.dumps(prediction), 'application/json'