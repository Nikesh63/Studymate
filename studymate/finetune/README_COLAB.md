# Fine-Tuning on Google Colab (No Local GPU Needed)

See earlier guidance: fine-tune Qwen2.5-1.5B-Instruct with Unsloth+QLoRA on
Colab's free T4 GPU, export to GGUF with model.save_pretrained_gguf(...),
then `ollama create my-finetuned-model -f Modelfile` to run it locally on
your CPU via Ollama. Point app/app.py's MODEL_NAME at "my-finetuned-model"
once that's done.
