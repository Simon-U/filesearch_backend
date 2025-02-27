import torch
import time
import logging
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import AutoProcessor, AutoModelForVision2Seq
from contextlib import contextmanager
import gc

# Configure logger
logger = logging.getLogger("image_processor.smolvlm")
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

class SmolVLMCaptioningBackend(BaseCaptioningBackend):
    """SmolVLM2 image captioning backend implementation"""
    
    def initialize(self):
        logger.info(f"Initializing SmolVLM captioning backend with model: {self.config.caption_model}")
        logger.info(f"Device: {self.config.device}, Half precision: {self.config.use_half_precision}")
        
        start_time = time.time()
        try:
            with torch_gc_context():
                model_kwargs = {
                    'cache_dir': self.config.cache_dir,
                    'torch_dtype': torch.bfloat16 if self.config.use_half_precision else torch.float32
                }
                
                if self.config.hf_token:
                    model_kwargs['token'] = self.config.hf_token
                    logger.info("Using provided HuggingFace token for model access")
                
                # Add flash attention if CUDA is available
                if torch.cuda.is_available():
                    model_kwargs['_attn_implementation'] = "flash_attention_2"
                    logger.info("Enabling flash attention for improved performance")
                
                logger.info(f"Loading processor for model: {self.config.caption_model}")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.caption_model,
                    **model_kwargs
                )
                
                logger.info(f"Loading model: {self.config.caption_model}")
                self.model = AutoModelForVision2Seq.from_pretrained(
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
        logger.info(f"SmolVLM captioning backend initialized in {elapsed_time:.2f} seconds")

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        start_time = time.time()
        logger.info(f"Starting image analysis. Image size: {image.size}, mode: {image.mode}")
        
        try:
            with torch_gc_context(), torch.no_grad():
                # Create input messages with appropriate prompt
                prompt_start_time = time.time()
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": """Describe this image in detail, focusing on all visible content and information. If this is a document:

                        Identify the type of document (report, map, diagram, slide, form, etc.)
                        Describe all visual elements present:

                        For graphs/charts: identify type, axes, data relationships, key values and trends
                        For maps: describe geographical area, legend items, marked locations, routes, or zones
                        For tables: outline structure, headers, and significant data points
                        For diagrams/flowcharts: explain components, connections, and processes shown
                        For forms/documents: note sections, fields, and any completed information



                        Extract all readable text including titles, labels, legends, annotations, and captions. Report numerical data accurately. Identify color-coding or visual emphasis if present. Note any scale indicators, north arrows, timestamps, or measurement units.
                        Summarize the main information being communicated and its significance. Be thorough yet concise, focusing on factual details present in the image."""}
                    ]
                }]
                
                logger.info("Applying chat template to prepare input prompt")
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                logger.debug(f"Generated prompt length: {len(prompt)} characters")
                
                logger.info("Processing image and converting to model inputs")
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Log input tensor shapes for debugging
                input_shapes = {k: v.shape for k, v in inputs.items() if hasattr(v, 'shape')}
                logger.info(f"Input tensor shapes: {input_shapes}")
                
                prompt_elapsed = time.time() - prompt_start_time
                logger.info(f"Input preparation completed in {prompt_elapsed:.2f} seconds")
                
                # Generate image description
                logger.info(f"Generating caption with max_new_tokens={self.config.max_caption_length}")
                generation_start_time = time.time()
                
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.config.max_caption_length,
                    do_sample=False  # Use greedy decoding for deterministic output
                )
                
                generation_elapsed = time.time() - generation_start_time
                logger.info(f"Caption generation completed in {generation_elapsed:.2f} seconds")
                logger.info(f"Generated sequence length: {generated_ids.shape}")
                
                # Extract assistant's response
                decode_start_time = time.time()
                description = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                # Clean up the response to extract just the caption part
                # SmolVLM produces "Assistant: {caption}" format
                if description.startswith("Assistant:"):
                    description = description[len("Assistant:"):].strip()
                
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
                    "confidence": 1.0,  # SmolVLM doesn't provide confidence scores
                    "processing_time": {
                        "total": total_elapsed,
                        "prompt_preparation": prompt_elapsed,
                        "generation": generation_elapsed,
                        "decoding": decode_elapsed
                    }
                }
                logger.info(f"Output: {output} ")
                return output
        except Exception as e:
            logger.error(f"Error during image captioning: {str(e)}", exc_info=True)
            return {
                "caption": "",
                "error": str(e),
                "processing_time": time.time() - start_time
            }