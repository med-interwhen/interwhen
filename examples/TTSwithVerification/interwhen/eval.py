import json

path = r"D:\interwhen\examples\TTSwithVerification\interwhen\Outputs_TTS\medreason\20260619_234837\outputs.jsonl"

with open(path) as f:
    first = json.loads(f.readline())

# Look at what the output_text looks like
print("=== OUTPUT TEXT (first 500 chars) ===")
print(first.get("output_text", "MISSING")[:500])

print("\n=== ALL VALUES ===")
for k, v in first.items():
    if k != "output_text":
        print(f"{k}: {v}")