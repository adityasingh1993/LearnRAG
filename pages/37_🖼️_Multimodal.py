"""
🖼️ Multi-modal Embeddings (Text-to-Image)
Real-time CLIP demonstration using sentence-transformers.
"""
import streamlit as st
from embedding_utils.common import inject_custom_css, page_header
from embedding_utils.visualization import PLOTLY_LAYOUT, plot_embeddings_3d, reduce_dimensions, plot_comparison_bars

inject_custom_css()

page_header(
    "Multi-Modal Embeddings",
    "🖼️",
    "Live CLIP Demo",
    "Experience true multimodality! We are running a LIVE AI model (CLIP) that mathematically maps images and text into the exact same universe."
)

st.markdown("""
### 🤯 The Magic of CLIP
In 2021, OpenAI released **CLIP** (Contrastive Language-Image Pre-training). 
Instead of just training on text, it was trained on millions of **(Image, Caption)** pairs.
Because of this, you can type **any text** and provide **any image**, and see exactly how closely the model thinks they match mathematically.
""")

st.markdown("---")

@st.cache_resource(show_spinner="Loading CLIP Model (might take a minute on first run)...")
def load_clip():
    from sentence_transformers import SentenceTransformer
    # clip-ViT-B-32 is popular, lightweight, and supported out of the box by sentence-transformers
    return SentenceTransformer('clip-ViT-B-32')

try:
    with st.spinner("Initializing AI Core..."):
        model = load_clip()
        clip_loaded = True
except Exception as e:
    st.error(f"Failed to load CLIP. (Are required libraries like Pillow installed?) Error: {e}")
    clip_loaded = False

if clip_loaded:
    st.markdown("### 📸 Live Image-to-Text Zero-Shot Search")
    st.write("Upload an image (or provide a URL) and enter several descriptions. CLIP will calculate the exact vector distance and rank which text best describes the image!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 1. Provide an Image")
        img_source = st.radio("Image Source:", ["URL", "Upload File"], horizontal=True)
        img = None
        if img_source == "Upload File":
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file).convert("RGB")
        else:
            url = st.text_input("Image URL:", value="https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400")
            if url:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content)).convert("RGB")
                    else:
                        st.error("Could not fetch the image from that URL.")
                except Exception as e:
                    st.error("Invalid URL or connection issue.")
        
        if img is not None:
            st.image(img, caption="Target Image", use_container_width=True)
            
    with col2:
        st.markdown("#### 2. Provide Custom Search Queries")
        st.write("Enter text descriptions (one per line). The model calculates the Cosine Similarity between your image vector and each text vector!")
        default_labels = "A cute dog\\nA fast sports car\\nA cup of hot coffee\\nA fluffy golden retriever\\nA wild wolf in the forest\\nA cat sleeping on a mat"
        labels_txt = st.text_area("Categories / Descriptions:", value=default_labels, height=180)
        labels = [l.strip() for l in labels_txt.split("\\n") if l.strip()]
        
        if img is not None and labels:
            if st.button("🔍 Run Live Inference", type="primary"):
                with st.spinner("Embedding Image & Text in the 512-dimensional Space..."):
                    # 1. Embed Image into Vector
                    img_emb = model.encode([img])[0]
                    # 2. Embed Texts into Vectors
                    txt_embs = model.encode(labels)
                    # 3. Calculate Cosine Similarity
                    similarities = cosine_similarity([img_emb], txt_embs)[0]
                    
                    results = sorted(zip(labels, similarities, txt_embs), key=lambda x: x[1], reverse=True)
                    
                st.markdown("#### 🏆 Similarity Rankings:")
                for rank, (label, score, _) in enumerate(results):
                    if rank == 0:
                        st.success(f"**🥇 {label}** (Similarity: {score:.3f})")
                    else:
                        st.info(f"**{rank+1}.** {label} (Similarity: {score:.3f})")
                        
                # 4. Visual Bar Chart
                fig = px.bar(
                    x=[r[1] for r in results][::-1], 
                    y=[r[0] for r in results][::-1],
                    orientation='h',
                    labels={'x': 'Cosine Similarity Score', 'y': ''},
                    color=[r[1] for r in results][::-1],
                    color_continuous_scale="Purples"
                )
                custom_layout = PLOTLY_LAYOUT.copy()
                custom_layout["height"] = 300 + (len(labels) * 20)
                custom_layout["margin"] = dict(l=20, r=20, t=10, b=20)
                fig.update_layout(**custom_layout)
                st.plotly_chart(fig, use_container_width=True)
                
                # ─── Under the Hood Visualization ───
                st.markdown("---")
                st.markdown("### 🧠 Under the Hood: Visualizing Multimodal AI")
                st.write("How does CLIP know that the image matches the text? Let's literally peek at the math running under the hood.")
                
                vis_tab1, vis_tab2 = st.tabs(["📊 1. Raw Vector Matching", "🌌 2. Shared 3D Universe"])
                
                with vis_tab1:
                    best_txt_emb = results[0][2]
                    best_txt_label = results[0][0]
                    
                    st.markdown(f"#### The Image Vector vs. The Winning Text Prompt: *\"{best_txt_label}\"*")
                    st.write(f"The CLIP model takes your image and turns it into a list of 512 numbers (the image vector). It does the exact same process for the text prompt. Notice how their numerical arrays literally mirror each other!")
                    
                    # Display the first 60 dims so it's readable
                    dims_to_show = min(60, len(img_emb))
                    features = [f"dim_{i}" for i in range(dims_to_show)]
                    
                    fig_bars = plot_comparison_bars(
                        bow_vector=img_emb[:dims_to_show],
                        tfidf_vector=best_txt_emb[:dims_to_show],
                        feature_names=features,
                        title=f"Image vs '{best_txt_label}' Vector Arrays (First {dims_to_show} Dims)"
                    )
                    # Hack existing function trace names in Plotly figure
                    fig_bars.data[0].name = "🖼️ Image Vector"
                    fig_bars.data[1].name = f"📝 Text Vector"
                    st.plotly_chart(fig_bars, use_container_width=True)
                
                with vis_tab2:
                    st.markdown("#### The Complete Map")
                    st.write(f"We mapped the Image vector and ALL your Text query vectors down to 3 dimensions using PCA mathematics. Notice how the image mathematically 'floats' closest to the text prompt that describes it best!")
                    
                    # Prepare embeddings matrix
                    all_embeddings = np.vstack([img_emb, txt_embs])
                    all_labels = ["⭐ TARGET IMAGE"] + [f"{l}" for l in labels]
                    all_colors = ["TARGET IMAGE"] + ["Text Prompt"] * len(labels)
                    
                    # Reduce to 3D
                    # For a tiny dataset, PCA is mathematically perfect for preserving relative distances
                    reduced = reduce_dimensions(all_embeddings, method="pca", n_components=3)
                    
                    if reduced.shape[1] < 3:
                        padding = np.zeros((reduced.shape[0], 3 - reduced.shape[1]))
                        reduced = np.hstack((reduced, padding))
                        
                    fig_3d = plot_embeddings_3d(
                        reduced=reduced,
                        labels=all_labels,
                        title="Image & Texts in Shared 3D Space",
                        color_labels=all_colors
                    )
                    
                    # Increase marker size and change symbol for the image
                    fig_3d.update_traces(marker=dict(size=9, line=dict(width=1, color="#FAFAFA")))
                    st.plotly_chart(fig_3d, use_container_width=True)
