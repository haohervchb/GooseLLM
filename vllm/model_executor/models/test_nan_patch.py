import re

file_path = "/home/rah/GooseLLM/vllm/model_executor/models/qwen3.py"
with open(file_path, "r") as f:
    text = f.read()

# I will replace the forward pass in Qwen3DecoderLayer
orig = """        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )"""

new = """        if torch.isnan(hidden_states).any():
            print(f"[DEBUG] Qwen3DecoderLayer input to self_attn has NaNs!")
        
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        
        if torch.isnan(hidden_states).any():
            print(f"[DEBUG] Qwen3DecoderLayer output from self_attn has NaNs! positions={positions}")"""

text = text.replace(orig, new)

orig2 = """        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual"""

new2 = """        hidden_states = self.mlp(hidden_states)
        if torch.isnan(hidden_states).any():
            print(f"[DEBUG] Qwen3DecoderLayer output from mlp has NaNs!")
        return hidden_states, residual"""

text = text.replace(orig2, new2)

with open(file_path, "w") as f:
    f.write(text)

print("Patched qwen3.py successfully.")
