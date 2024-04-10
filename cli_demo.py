import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "Nobitaxi/InternLM2-chat-7B-SQL"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """If you are an expert in SQL, please generate a good SQL Query for Question based on the CREATE TABLE statement."""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM2-chat-7b-sql chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
