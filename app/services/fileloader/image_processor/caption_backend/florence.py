import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import AutoProcessor, AutoModelForCausalLM
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.florence")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

@contextmanager
def torch_gc_context():
    """Context manager for proper CUDA memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

class FlorenceCaptioningBackend(BaseCaptioningBackend):
    """Florence-2 image captioning backend implementation"""
    
    def initialize(self):
        logger.info(f"Initializing Florence captioning backend with model: {self.config.caption_model}")
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                model_kwargs = {
                    'cache_dir': self.config.cache_dir,
                    'torch_dtype': torch.float16 if self.config.use_half_precision else torch.float32,
                    'trust_remote_code': True  # Required for Florence model
                }
                
                if self.config.hf_token:
                    model_kwargs['token'] = self.config.hf_token
                    logger.info("Using provided HuggingFace token for model access")
                
                logger.info(f"Loading processor for model: {self.config.caption_model}")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.caption_model,
                    trust_remote_code=True,
                    token=self.config.hf_token if self.config.hf_token else None
                )
                
                logger.info(f"Loading model: {self.config.caption_model}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.caption_model,
                    **model_kwargs
                ).to(self.config.device)
                
                if self.config.optimize_memory_usage:
                    logger.info("Setting model to evaluation mode for memory optimization")
                    self.model.eval()
                
                # Log model size and memory usage
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                # Log number of parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"Model loaded with {total_params:,} parameters")
        except Exception as e:
            logger.error(f"Error during model initialization: {str(e)}", exc_info=True)
            raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"Florence captioning backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting image analysis. Image size: {image.size}, mode: {image.mode}")
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Use MORE_DETAILED_CAPTION prompt for richer descriptions
                prompt_start_time = time.time()
                prompt = "<MORE_DETAILED_CAPTION>"
                
                logger.info("Processing image with Florence model")
                inputs = self.processor(
                    text=prompt, 
                    images=image, 
                    return_tensors="pt"
                )
                inputs = self.to_device(inputs)
                
                # Log input tensor shapes for debugging
                input_shapes = {k: v.shape for k, v in inputs.items() if hasattr(v, 'shape')}
                logger.info(f"Input tensor shapes: {input_shapes}")
                
                prompt_elapsed = time.time() - prompt_start_time
                logger.info(f"Input preparation completed in {prompt_elapsed:.2f} seconds")
                
                # Generate detailed image description
                logger.info(f"Generating caption with max_new_tokens={self.config.max_caption_length}")
                generation_start_time = time.time()
                
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=self.config.max_caption_length,
                    num_beams=3,  # Use beam search for better quality
                    do_sample=False  # Use deterministic generation
                )
                
                generation_elapsed = time.time() - generation_start_time
                logger.info(f"Caption generation completed in {generation_elapsed:.2f} seconds")
                
                # Extract and process the generated text
                decode_start_time = time.time()
                generated_text = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=False
                )[0]
                
                # Post-process the output to extract the actual caption
                # Florence uses the post_process_generation method
                try:
                    parsed_answer = self.processor.post_process_generation(
                        generated_text, 
                        task=prompt, 
                        image_size=(image.width, image.height)
                    )
                    
                    # The parsed_answer should be a dict with <MORE_DETAILED_CAPTION> as key
                    if isinstance(parsed_answer, dict) and "<MORE_DETAILED_CAPTION>" in parsed_answer:
                        description = parsed_answer["<MORE_DETAILED_CAPTION>"]
                    else:
                        # Fallback to simple text extraction
                        description = str(parsed_answer)
                except Exception as proc_error:
                    logger.warning(f"Error in post-processing: {proc_error}")
                    # Fallback to simple text extraction if post-processing fails
                    description = generated_text.replace("<MORE_DETAILED_CAPTION>", "").strip()
                
                decode_elapsed = time.time() - decode_start_time
                logger.info(f"Decoding and post-processing completed in {decode_elapsed:.2f} seconds")
                logger.info(f"Final caption length: {len(description)} characters")
                
                # Log memory usage after generation
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                    memory_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    logger.info(f"CUDA memory allocated: {memory_allocated:.2f} MB")
                    logger.info(f"CUDA memory reserved: {memory_reserved:.2f} MB")
                
                total_elapsed = time.time() - start_time
                logger.info(f"Total processing time: {total_elapsed:.2f} seconds")
                
                output = {
                    "caption": description,
                    "confidence": 1.0,  # Florence doesn't provide confidence scores
                    "processing_time": {
                        "total": total_elapsed,
                        "prompt_preparation": prompt_elapsed,
                        "generation": generation_elapsed,
                        "decoding": decode_elapsed
                    }
                }
                return output
                
        except Exception as e:
            logger.error(f"Error during image captioning: {str(e)}", exc_info=True)
            return {
                "caption": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }