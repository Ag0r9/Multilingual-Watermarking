from transformers import LogitsProcessor

VOCAB_SIZE = 50257

class RandomHashLogitsProcessor(LogitsProcessor):
    def __init__(self, hash_function, vocab_size=VOCAB_SIZE):
        self.hash_function = hash_function
        self.vocab_size = vocab_size

    def __call__(self, input_ids, logits, **kwargs):
        random_hash = self.hash_function(input_ids, self.vocab_size)
        modified_logits = logits + random_hash
        return modified_logits
    
    def __repr__(self):
        return f"{self.__class__.__name__}(hash_function={self.hash_function})"
    
    def __len__(self):
        return 0