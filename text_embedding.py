from transformers import AutoTokenizer, T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = T5EncoderModel.from_pretrained("t5-small")

text = "Studies have been shown that owning a dog is good for you"
input_ids = tokenizer(text, return_tensors="pt").input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
last_hidden_states = outputs.last_hidden_state

print(last_hidden_states.shape)


