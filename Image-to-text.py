from transformers import pipeline

pipe = pipeline("image-to-text", model="microsoft/git-base")

# Load model directly
from transformers import AutoProcessor, AutoModelForCausalLM

processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")
