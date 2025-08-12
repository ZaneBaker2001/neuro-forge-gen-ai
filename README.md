# GenAI Studio — An Impressive Generative AI Project

GenAI Studio is a **full-featured, local-first** generative AI playground:
- 🧠 **Text generation & chat** using Hugging Face `transformers` (with optional OpenAI API fallback).
- 🖼️ **Image generation** using `diffusers` (Stable Diffusion) with safety filtering.
- 📚 **RAG (Retrieval-Augmented Generation)** over your own files using `sentence-transformers` + `faiss`.
- 🧪 **Prompt Lab** to design, save, and test prompt templates (Jinja2-style variables).
- 🎛️ Built as a **Streamlit** app with clean UI and modular Python package (`genlab`).

> You can run everything locally on CPU (slow but works) or enable GPU acceleration. OpenAI is optional.

---

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2) Install requirements
```bash
pip install -r requirements.txt
```

> If you want to use OpenAI as a fallback for text generation, set an environment variable:
> ```bash
> export OPENAI_API_KEY="sk-..."
> ```

### 3) Run the app
```bash
streamlit run streamlit_app.py
```

Then open the local URL printed by Streamlit.

---

## Features

### Text Generation (Chat)
- Choose **local HF model** (default: a small instruct model) or **OpenAI** fallback.
- Adjustable **temperature**, **max tokens**, and **system prompt**.
- Simple conversation memory in-session.

### Image Generation
- Stable Diffusion (via `diffusers`) with:
  - Adjustable **guidance scale** and **inference steps**,
  - **Seed** for reproducibility,
  - Built-in **NSFW safety checker** (can be disabled if you understand the risks).

### RAG (Bring Your Own Knowledge)
- Drag & drop files (txt, md, pdf*) to create a vector index (FAISS).
- Query with **retrieval-augmented** answers using the chosen LLM.
- *PDF support requires `pypdf`.*

### Prompt Lab
- Create, edit, and save **Jinja2-like prompt templates** with variables.
- Test prompts live against your LLM.

---

## Project Structure

```
genai-studio/
├── genlab/
│   ├── __init__.py
│   ├── text_gen.py
│   ├── image_gen.py
│   ├── rag.py
│   └── utils.py
├── sample_data/
│   └── sample.txt
├── tests/
│   └── test_utils.py
├── requirements.txt
├── streamlit_app.py
└── README.md
```

---

## Notes & Tips

- **Models**: By default, we use lightweight defaults to lower the barrier to entry. You can change the model names in the UI.
- **Performance**: CPU works but is slow. If you have a GPU, install the appropriate PyTorch build (CUDA) for the best experience.
- **Privacy**: Keep the **OpenAI toggle off** if you want to ensure fully local processing.
- **Safety**: The image pipeline includes a safety checker by default; disabling it may generate unsafe content.
- **Licenses**: Respect the licenses and usage policies of any model you download.

---

## License

MIT
