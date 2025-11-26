from transformers import AutoModelForCausalLM, AutoTokenizer

prompt = "Alice and Bob"
checkpoint = "facebook/layerskip-llama3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(checkpoint)
outputs = model.generate(**inputs, assistant_early_exit=4, do_sample=False, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)