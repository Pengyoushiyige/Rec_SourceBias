import torch
from transformers import AutoModel, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

print("Loading model...")
model = AutoModel.from_pretrained('bert-base-uncased')
model = model.to(device)

print("Encoding text...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
inputs = tokenizer("Hello world", return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)
print("Output shape:", outputs.last_hidden_state.shape)
print("Success!")
