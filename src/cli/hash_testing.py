import typer
from src.hashes.RandomHashLogitsProcessor import RandomHashLogitsProcessor
from src.hashes.hash_functions import random_hash_function
from transformers import AutoTokenizer, AutoModelForCausalLM

app = typer.Typer()

@app.command()
def test_random_hash_function(prompt: str = "What's the meaning of life?", VOCAB_SIZE: int = 50257):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("gpt2", vocab_size=VOCAB_SIZE)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    # Create an instance of the RandomHashLogitsProcessor
    logits_processor = RandomHashLogitsProcessor(hash_function=random_hash_function)    
    inputs = tokenizer(prompt, return_tensors="pt").input_ids
    # Generate predictions with modified logits
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        logits_processor=logits_processor
    )
    # Decode the outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded_outputs)

if __name__ == "__main__":
    app()