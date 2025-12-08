import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, record_function, ProfilerActivity, schedule

checkpoint = "facebook/layerskip-llama3.2-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model on {device}...")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 设置 Schedule
# wait=1, warmup=1, active=1, repeat=1 -> 总共跑 3 步
my_schedule = schedule(wait=1, warmup=1, active=1, repeat=1)

# 定义一个回调函数：当 active 阶段结束时，自动保存 JSON
def trace_handler(p):
    output_file = "llama3_trace.json"
    p.export_chrome_trace(output_file)
    print(f"\n✅ Profile 完成！结果已自动保存为: {output_file}")

print("Starting Profiling...")

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=my_schedule,
    # 【修复点】这里直接绑定上面的 handler，让它自动保存
    on_trace_ready=trace_handler, 
    record_shapes=True,
    profile_memory=True,
    with_stack=True 
) as prof:
    
    # 循环 3 次以配合 schedule (Wait -> Warmup -> Active)
    for step in range(3):
        print(f"Step {step}: {'Wait' if step==0 else 'Warmup' if step==1 else 'Recording'}")
        
        with record_function("model_inference_step"):
            model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        prof.step()

# 【修复点】删除了最后的手动 export，防止重复保存报错