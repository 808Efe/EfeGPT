from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the pad_token to be the same as eos_token (since GPT-2 does not have a pad token)
tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loop this part so user can prompt more than one time
while(True):

    # Input text
    input_text = input("Enter your prompt : ")

    # Tokenize input
    inputs = tokenizer(
        input_text,
        padding="max_length",  # Pad to max length for consistency
        truncation=True,       # Truncate to max length
        max_length=1024,
        return_attention_mask=True
        
    )

    # Move input tensors to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Print the device of input tensors
    print(f"\nInput tensors are on: {inputs['input_ids'].device}")

    # Generate text without updating model gradients(no training)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        num_return_sequences=1,
        temperature=1.0,
        repetition_penalty=1.2
        
    )

    # Decode the tokenized output back into human-readable text, skipping special tokens (padding or end-of-sequence tokens)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
 
    # Print the decoded output
    print(f"\nResponse : {output_text}\n")