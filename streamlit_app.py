import io
import os
import base64
import streamlit as st
from PIL import Image

from gen.text_gen import TextGenerator, ChatMessage
from gen.image_gen import ImageGenerator, ImageConfig
from gen.rag import RAG, RAGConfig
from gen.utils import GenerationConfig, device_str, set_seed

st.set_page_config(page_title="GenAI Studio", page_icon="🤖", layout="wide")

st.sidebar.title("GenAI Studio")
st.sidebar.caption("Local-first generative AI playground")

# Global controls
with st.sidebar.expander("Compute & Models", expanded=True):
    use_openai = st.toggle("Use OpenAI for text", value=False, help="If enabled, requires OPENAI_API_KEY environment variable.")
    hf_model = st.text_input("HF chat model", value="HuggingFaceH4/zephyr-7b-alpha", help="Change to any chat-capable model you have access to.")
    sd_model = st.text_input("Stable Diffusion model", value="runwayml/stable-diffusion-v1-5")
    seed_value = st.number_input("Global seed (optional)", min_value=0, max_value=2**32-1, value=0, help="0 means random seed each run.")

if seed_value != 0:
    os.environ["GENAI_STUDIO_SEED"] = str(int(seed_value))
else:
    os.environ.pop("GENAI_STUDIO_SEED", None)

st.sidebar.write(f"Device: **{device_str()}**")

tabs = st.tabs(["💬 Chat", "🖼️ Image", "📚 RAG", "🧪 Prompt Lab", "ℹ️ About"])

# ---------------- Chat Tab ----------------
with tabs[0]:
    st.header("💬 Chat")
    st.caption("Chat with a local HF model or OpenAI fallback.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ChatMessage(role="system", content="You are GenAI Studio, a helpful assistant.")]

    colc1, colc2 = st.columns([3,1])
    with colc1:
        system_prompt = st.text_area("System prompt", value=st.session_state.chat_history[0].content, height=100)
    with colc2:
        temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
        max_new = st.slider("Max new tokens", 32, 1024, 256, 32)
    st.session_state.chat_history[0] = ChatMessage(role="system", content=system_prompt)

    user_msg = st.text_area("Your message", value="", placeholder="Ask me anything...")
    if st.button("Send", use_container_width=True):
        if user_msg.strip():
            st.session_state.chat_history.append(ChatMessage(role="user", content=user_msg.strip()))
            tg = TextGenerator(use_openai=use_openai, hf_model=hf_model)
            with st.spinner("Generating..."):
                reply = tg.generate(st.session_state.chat_history, GenerationConfig(temperature=temp, max_new_tokens=max_new))
            st.session_state.chat_history.append(ChatMessage(role="assistant", content=reply))

    for msg in st.session_state.chat_history[1:]:
        if msg.role == "user":
            st.chat_message("user").write(msg.content)
        elif msg.role == "assistant":
            st.chat_message("assistant").write(msg.content)

# ---------------- Image Tab ----------------
with tabs[1]:
    st.header("🖼️ Image Generation")
    st.caption("Stable Diffusion via diffusers")

    prompt = st.text_area("Prompt", value="A cozy reading nook with warm sunlight, ultra-detailed, cinematic lighting", height=100)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        steps = st.slider("Steps", 10, 60, 30, 1)
    with c2:
        guidance = st.slider("Guidance", 1.0, 15.0, 7.5, 0.5)
    with c3:
        width = st.selectbox("Width", [384, 448, 512, 640, 768], index=2)
    with c4:
        height = st.selectbox("Height", [384, 448, 512, 640, 768], index=2)

    c5, c6 = st.columns(2)
    with c5:
        seed = st.number_input("Seed (0 for random)", min_value=0, max_value=2**32-1, value=0)
    with c6:
        disable_safety = st.toggle("Disable safety checker (NSFW risk)", value=False)

    if st.button("Generate Image", use_container_width=True):
        if seed != 0:
            set_seed(seed)
        gen = ImageGenerator(model_name=sd_model)
        cfg = ImageConfig(steps=steps, guidance=guidance, width=width, height=height, seed=seed if seed != 0 else None, disable_safety=disable_safety)
        with st.spinner("Rendering..."):
            image, used_seed = gen.generate(prompt, cfg)
        st.image(image, caption=f"Seed: {used_seed}", use_column_width=True)
        # download
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        href = f'<a href="data:file/png;base64,{b64}" download="genai_studio.png">Download PNG</a>'
        st.markdown(href, unsafe_allow_html=True)

# ---------------- RAG Tab ----------------
with tabs[2]:
    st.header("📚 Retrieval-Augmented Generation")
    st.caption("Upload files, build a FAISS index, and query with your LLM.")

    if "rag" not in st.session_state:
        st.session_state.rag = None

    colr1, colr2 = st.columns([2,1])
    with colr1:
        top_k = st.slider("Top-K", 1, 10, 5, 1)
        embed_model = st.text_input("Embedding model", value="sentence-transformers/all-MiniLM-L6-v2")
    with colr2:
        if st.button("Initialize RAG"):
            st.session_state.rag = RAG(RAGConfig(embed_model=embed_model, top_k=top_k), text_gen=TextGenerator(use_openai=use_openai, hf_model=hf_model))
            st.success("RAG initialized.")

    if st.session_state.rag is not None:
        uploads = st.file_uploader("Upload files (txt, md, pdf)", type=["txt","md","pdf"], accept_multiple_files=True)
        if uploads:
            for f in uploads:
                st.session_state.rag.add_file(f.read(), f.name)
            st.success(f"Added {len(uploads)} file(s) to the index.")

        question = st.text_input("Ask a question about your files")
        if st.button("Ask", use_container_width=True):
            with st.spinner("Retrieving + Generating..."):
                answer, ctx = st.session_state.rag.query(question, top_k=top_k)
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Top Context Chunks")
            for i, (snippet, score) in enumerate(ctx, 1):
                st.markdown(f"**{i}.** {snippet} — _score: {score:.3f}_")

# ---------------- Prompt Lab Tab ----------------
with tabs[3]:
    st.header("🧪 Prompt Lab")
    st.caption("Create and test prompt templates with variables.")

    from jinja2 import Template

    name = st.text_input("Template name", value="example")
    raw_template = st.text_area("Template", value="You are a helpful assistant.\n\nWrite a short bio for {{ person }} who works as a {{ role }}.", height=160)
    vars_raw = st.text_area("Variables (JSON)", value='{"person":"Ada Lovelace","role":"mathematician"}', height=100)
    with st.expander("Generation settings"):
        temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, key="pl_temp")
        max_new = st.slider("Max new tokens", 32, 1024, 200, 32, key="pl_max")

    if st.button("Render + Generate", use_container_width=True):
        try:
            data = {} if not vars_raw.strip() else dict(**__import__("json").loads(vars_raw))
            prompt = Template(raw_template).render(**data)
            tg = TextGenerator(use_openai=use_openai, hf_model=hf_model)
            msg = [ChatMessage(role="system", content="Follow instructions faithfully."),
                   ChatMessage(role="user", content=prompt)]
            with st.spinner("Generating..."):
                out = tg.generate(msg, GenerationConfig(temperature=temp, max_new_tokens=max_new))
            st.subheader("Rendered Prompt")
            st.code(prompt)
            st.subheader("Model Output")
            st.write(out)
        except Exception as e:
            st.error(str(e))

# ---------------- About Tab ----------------
with tabs[4]:
    st.header("ℹ️ About")
    st.write("GenAI Studio is an open, local-first playground for text, image, and RAG workflows.")
    st.write("Swap models as you like and keep your data private. MIT licensed.")
    st.markdown("---")
    st.write("**Pro tip:** For best performance, install a CUDA-enabled PyTorch and use smaller models on laptops.")
