import torch
from typing import Dict, Any
from PIL import Image
from .base import BaseCaptioningBackend
from transformers import AutoProcessor, AutoModelForVision2Seq
from contextlib import contextmanager
import gc

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
        with torch_gc_context():
            model_kwargs = {
                'cache_dir': self.config.cache_dir,
                'torch_dtype': torch.bfloat16 if self.config.use_half_precision else torch.float32
            }
            
            if self.config.hf_token:
                model_kwargs['token'] = self.config.hf_token
            
            # Add flash attention if CUDA is available
            if torch.cuda.is_available():
                model_kwargs['_attn_implementation'] = "flash_attention_2"
            
            self.processor = AutoProcessor.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.config.caption_model,
                **model_kwargs
            ).to(self.config.device)
            
            if self.config.optimize_memory_usage:
                self.model.eval()

    def analyze(self, image: Image.Image) -> Dict[str, Any]:
        try:
            with torch_gc_context(), torch.no_grad():
                # Create input messages with appropriate prompt
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
                
                prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=prompt, images=[image], return_tensors="pt")
                inputs = self.to_device(inputs)
                
                # Generate image description
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.config.max_caption_length,
                    do_sample=False  # Use greedy decoding for deterministic output
                )
                
                # Extract assistant's response
                description = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0]
                
                # Clean up the response to extract just the caption part
                # SmolVLM produces "Assistant: {caption}" format
                if description.startswith("Assistant:"):
                    description = description[len("Assistant:"):].strip()
                
                return {
                    "caption": description,
                    "confidence": 1.0  # SmolVLM doesn't provide confidence scores
                }
        except Exception as e:
            return {
                "caption": "",
                "error": str(e)
            }