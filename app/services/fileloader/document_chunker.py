import spacy
import re
import html
import unicodedata
from transformers import AutoTokenizer
from bs4 import BeautifulSoup

__all__ = ['DocumentChunker']

LANG_MODELS = {
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm"
}

class DocumentChunker:
    def __init__(self, embed_model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", language='en'):
        self.tokenizer = AutoTokenizer.from_pretrained(embed_model_id)

        # Load a multilingual spaCy model
        # Load spaCy model or create a blank one if unavailable
        model_name = LANG_MODELS.get(language, None)
        if model_name:
            try:
                self.nlp = spacy.load(model_name)
            except OSError:
                print(f"Model '{model_name}' not found. Installing...")
                spacy.cli.download(model_name)
                self.nlp = spacy.load(model_name)
        else:
            print(f"Language '{language}' not supported by spaCy models. Using a blank model.")
            self.nlp = spacy.blank(language)

    @staticmethod
    def preprocess_document(doc):
        """
        Cleans a document while preserving meaningful content.
        
        :param doc: The input document string containing HTML and special characters.
        :return: Cleaned document string.
        """
        # First pass: Handle HTML content
        try:
            # Parse HTML and extract text
            soup = BeautifulSoup(doc, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            doc = soup.get_text(separator=' ')
        except Exception as e:
            # If HTML parsing fails, try direct regex replacement
            doc = re.sub(r'<[^>]+>', ' ', doc)

        # Remove Markdown headers, bullet points, and excessive symbols
        doc = re.sub(r"[#*•◦-]+", " ", doc)  # Remove Markdown symbols and bullets
        doc = re.sub(r"[=_\-]{3,}", " ", doc)  # Remove separators (---, ===)
        doc = re.sub(r"[.]{3,}", ".", doc)  # Replace long ellipses
        doc = re.sub(r"\|", " ", doc)  # Convert table bars to spaces
        doc = re.sub(r"\s+", " ", doc).strip()  # Normalize spaces

        # Remove HTML entities and special characters
        doc = html.unescape(doc)
        doc = re.sub(r'&#\d+;', ' ', doc)

        # Remove common font names and formatting text
        #formatting_terms = [
        #    "times new roman", "calibri", "arial", "courier new", "verdana", 
        #    "font-family", "font size", "bold", "italic", "underline",
        #    "left-aligned", "right-aligned", "justified", "centered"
        #]
        
        #pattern = r"\b(" + "|".join(formatting_terms) + r")\b|\b(?:[A-Z][a-z]+){2,}\b"
        #doc = re.sub(pattern, " ", doc, flags=re.IGNORECASE)
        # Remove placeholders
        doc = re.sub(r"<missing-text>|<redacted>|<confidential>", "", doc)
        # Remove URLs while preserving company names
        doc = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', doc)
        
        # Remove specific boilerplate content
        patterns_to_remove = [
            
            # Additional cleanups
            r'::\s*',  # Remove double colons with spaces
            r'\s*::\s*',  # Remove double colons with optional spaces
        ]
        
        for pattern in patterns_to_remove:
            doc = re.sub(pattern, '', doc, flags=re.IGNORECASE | re.MULTILINE)

        # Keep punctuation that affects meaning
        doc = re.sub(r'[^\w\s\-.,;:?!%$"\'()]', ' ', doc)
        
        # Normalize whitespace
        doc = re.sub(r'\s+', ' ', doc).strip()
        
        # Remove empty parentheses and brackets
        doc = re.sub(r'\(\s*\)', '', doc)
        doc = re.sub(r'\[\s*\]', '', doc)
        
        # Remove any lines that are just numbers or very short
        #doc = '\n'.join(line for line in doc.split('\n') 
        #            if len(line.strip()) > 5 and not line.strip().isdigit())
        return doc.strip()




    def chunk_doc(self, doc, max_token=512, overlap_ratio=0.1, skip=1):
        """
        Splits a document into chunks, keeping full sentences and maintaining token limits.

        :param doc: The document text to split.
        :param max_token: Maximum token length per chunk.
        :param overlap_ratio: Fraction of chunk to overlap (default: 10%).
        :param skip: Skip strategy (1 = no skip, 2 = every second chunk, etc.).
        :return: List of document chunks.
        """
        # Step 1: Preprocess the document (clean formatting, normalize text)
        doc = self.preprocess_document(doc)
        # Step 2: Fix split words (reconnect broken words from PDF extraction)

        #doc = self.fix_split_words_spacy(doc)
        # Step 3: Sentence Tokenization
        doc = self.nlp(doc)
        sentences = [sent.text for sent in doc.sents]

        # Step 4: Create Chunks While Keeping Sentences Intact
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            token_length = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            if current_length + token_length <= max_token:
                # Add sentence to the current chunk
                current_chunk.append(sentence)
                current_length += token_length
            else:
                # If chunk is full, finalize it
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = token_length

        # Add last chunk if it exists
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Step 5: Apply Overlap to Maintain Context
        overlap_size = int(len(chunks) * overlap_ratio)
        final_chunks = []

        for i in range(len(chunks)):
            if i > 0:
                # Include overlap from the previous chunk
                overlap = chunks[max(0, i - overlap_size):i]
                chunk_with_overlap = " ".join(overlap) + " " + chunks[i]
            else:
                chunk_with_overlap = chunks[i]

            if chunk_with_overlap.strip():  # Ensure no empty chunks
                final_chunks.append(chunk_with_overlap)

        # Step 6: Apply Skip Strategy (Optional)
        return final_chunks[::skip] if skip > 1 else final_chunks
