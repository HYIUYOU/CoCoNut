from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import layer_time
prompts = [
    "What is the role of gravity in the solar system?",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.1,max_tokens=1)


llm = LLM(model="/root/heyiyuan/CoCoNut/Models/Llama-2-7b-chat-hf")

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("=="*50)

print(len(layer_time))
print(len(layer_time[0]))
print(len(layer_time[1]))
print(len(layer_time[10]))
print(len(layer_time[21]))
print(len(layer_time[30]))
# print(layer_time[10])
# print(layer_time[11])
print("=="*50)