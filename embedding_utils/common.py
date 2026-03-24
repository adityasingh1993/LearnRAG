"""
Common utilities: sample data, UI helpers, and shared components.
"""
import streamlit as st

# ─── Sample Sentences (grouped by topic for clustering demos) ───
SAMPLE_SENTENCES = {
    "Animals": [
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "The kitten played with yarn",
        "Puppies love to fetch sticks",
        "Birds fly high in the sky",
    ],
    "Technology": [
        "Artificial intelligence is transforming industries",
        "Machine learning models need large datasets",
        "Neural networks mimic the human brain",
        "Deep learning enables image recognition",
        "Computers process data at lightning speed",
    ],
    "Food": [
        "Pizza is a popular Italian dish",
        "Sushi is made with raw fish and rice",
        "Chocolate cake is a delicious dessert",
        "Fresh fruits are healthy snacks",
        "Coffee keeps people awake in the morning",
    ],
    "Sports": [
        "Football is the most popular sport worldwide",
        "Basketball requires speed and agility",
        "Swimming is excellent exercise",
        "Tennis matches can last for hours",
        "Running a marathon tests endurance",
    ],
}

FLAT_SAMPLES = []
SAMPLE_LABELS = []
SAMPLE_COLORS = []
COLOR_MAP = {
    "Animals": "#FF6B6B",
    "Technology": "#6C63FF",
    "Food": "#4ECB71",
    "Sports": "#FFB347",
}
for category, sentences in SAMPLE_SENTENCES.items():
    for s in sentences:
        FLAT_SAMPLES.append(s)
        SAMPLE_LABELS.append(category)
        SAMPLE_COLORS.append(COLOR_MAP[category])

# ─── Polysemy examples for contextual embedding demos ───
POLYSEMY_EXAMPLES = [
    {
        "word": "bank",
        "sentence_a": "I deposited money at the bank",
        "sentence_b": "We sat by the river bank",
        "context_a": "Financial",
        "context_b": "Nature",
    },
    {
        "word": "bat",
        "sentence_a": "He hit the ball with a bat",
        "sentence_b": "A bat flew out of the cave",
        "context_a": "Sports",
        "context_b": "Animal",
    },
    {
        "word": "crane",
        "sentence_a": "The construction crane lifted the beam",
        "sentence_b": "A crane stood in the shallow water",
        "context_a": "Machine",
        "context_b": "Bird",
    },
    {
        "word": "spring",
        "sentence_a": "Flowers bloom in spring",
        "sentence_b": "The spring in the mattress broke",
        "context_a": "Season",
        "context_b": "Mechanism",
    },
]

# ─── Timeline Data ───
TIMELINE = [
    {
        "year": "1954",
        "name": "Bag of Words",
        "icon": "📊",
        "desc": "Simple word count vectors — the foundation of text representation.",
        "page": "1_📊_Bag_of_Words",
    },
    {
        "year": "1972",
        "name": "TF-IDF",
        "icon": "📈",
        "desc": "Weighs terms by importance — common words matter less.",
        "page": "2_📈_TF_IDF",
    },
    {
        "year": "2013",
        "name": "Word2Vec",
        "icon": "🧠",
        "desc": "Dense learned vectors — words with similar meanings cluster together.",
        "page": "3_🧠_Word2Vec",
    },
    {
        "year": "2014",
        "name": "GloVe",
        "icon": "🌐",
        "desc": "Global co-occurrence statistics — combining the best of count-based and prediction-based methods.",
        "page": "4_🌐_GloVe",
    },
    {
        "year": "2018",
        "name": "Transformers",
        "icon": "🤖",
        "desc": "Contextual embeddings — same word, different meaning in different contexts.",
        "page": "6_🤖_Transformers",
    },
]


# ─── UI Components ───

def page_config(title: str):
    """Set common page config."""
    st.set_page_config(
        page_title=f"Embeddings | {title}",
        page_icon="🔤",
        layout="wide",
    )


def page_header(title: str, icon: str, year: str, description: str):
    """Render a consistent page header."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1d23 0%, #2d1f4e 100%);
            border-radius: 16px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(108, 99, 255, 0.3);
            box-shadow: 0 8px 32px rgba(108, 99, 255, 0.15);
        ">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;">
                <span style="font-size: 2.5rem;">{icon}</span>
                <div>
                    <h1 style="margin:0; font-size: 2rem; background: linear-gradient(90deg, #6C63FF, #B794F6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                        {title}
                    </h1>
                    <span style="
                        background: rgba(108, 99, 255, 0.2);
                        color: #B794F6;
                        padding: 0.2rem 0.8rem;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: 600;
                    ">📅 {year}</span>
                </div>
            </div>
            <p style="color: #a0aec0; margin: 0.8rem 0 0 0; font-size: 1.1rem; line-height: 1.5;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_pros_cons(pros: list, cons: list):
    """Display strengths and weaknesses in a styled card."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """<div style="
                background: rgba(78, 203, 113, 0.08);
                border: 1px solid rgba(78, 203, 113, 0.3);
                border-radius: 12px;
                padding: 1.2rem;
            "><h4 style="color: #4ECB71; margin-top:0;">✅ Strengths</h4>""",
            unsafe_allow_html=True,
        )
        for p in pros:
            st.markdown(f"- {p}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(
            """<div style="
                background: rgba(255, 107, 107, 0.08);
                border: 1px solid rgba(255, 107, 107, 0.3);
                border-radius: 12px;
                padding: 1.2rem;
            "><h4 style="color: #FF6B6B; margin-top:0;">⚠️ Weaknesses</h4>""",
            unsafe_allow_html=True,
        )
        for c in cons:
            st.markdown(f"- {c}")
        st.markdown("</div>", unsafe_allow_html=True)


def metric_card(label: str, value: str, description: str = ""):
    """Render a styled metric card."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1a1d23, #252830);
            border: 1px solid rgba(108, 99, 255, 0.2);
            border-radius: 12px;
            padding: 1.2rem;
            text-align: center;
        ">
            <div style="color: #6C63FF; font-size: 1.8rem; font-weight: 700;">{value}</div>
            <div style="color: #FAFAFA; font-size: 0.95rem; font-weight: 600; margin-top: 0.3rem;">{label}</div>
            <div style="color: #718096; font-size: 0.8rem; margin-top: 0.2rem;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_custom_css():
    """Inject global CSS for polished look."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0E1117 0%, #1A1D23 100%);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(108, 99, 255, 0.1);
            border-radius: 8px;
        }

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(108, 99, 255, 0.1);
            border-radius: 8px 8px 0 0;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(108, 99, 255, 0.3);
        }

        /* Metric card animations */
        div[data-testid="stMetric"] {
            background: rgba(108, 99, 255, 0.05);
            border: 1px solid rgba(108, 99, 255, 0.2);
            border-radius: 12px;
            padding: 1rem;
        }

        /* Dataframe styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }

        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, #6C63FF, #B794F6);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);
        }

        /* Code block styling */
        .stCodeBlock {
            border-radius: 12px;
        }

        /* Hide hamburger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
