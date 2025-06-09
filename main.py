import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from PIL import Image
import requests
from io import BytesIO
import os
from pathlib import Path
import logging
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MUST BE FIRST: Configure Streamlit page
st.set_page_config(
    page_title="Document Intelligence Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-message {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .response-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .image-container {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #fafbfc;
    }
    
    .status-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2rem 0;
    }
    
    .language-selector {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class ProfessionalChatbot:
    def __init__(self):
        self.setup_genai()
        self.setup_vector_db()
        self.initialize_session_state()
        self.setup_language_prompts()
    
    def setup_genai(self):
        """Initialize Gemini AI model"""
        try:
            genai.configure(api_key="AIzaSyAvzloY_NyX-yjtZb8EE_RdXPs3rPmMEso")
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            logger.info("Gemini AI model initialized successfully")
        except Exception as e:
            st.error("Failed to initialize AI model. Please check your API key.")
            logger.error(f"Gemini AI initialization error: {e}")
            st.stop()
    
    def setup_vector_db(self):
        """Initialize vector database"""
        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db_path = "comprehensive_vector_db"
            
            if os.path.exists(self.vector_db_path):
                self.vector_db = Chroma(
                    persist_directory=self.vector_db_path, 
                    embedding_function=self.embedding_model
                )
                status_text = "✅ Knowledge base connected" if st.session_state.get('language', 'English') == 'English' else "✅ ナレッジベースに接続しました"
                st.markdown(
                    f'<div class="status-indicator status-success">{status_text}</div>', 
                    unsafe_allow_html=True
                )
                logger.info(f"Vector database loaded from {self.vector_db_path}")
            else:
                error_text = "Knowledge base not found. Please contact system administrator." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースが見つかりません。システム管理者にお問い合わせください。"
                st.error(error_text)
                logger.error(f"Vector database not found at {self.vector_db_path}")
                st.stop()
        except Exception as e:
            error_text = "Failed to connect to knowledge base." if st.session_state.get('language', 'English') == 'English' else "ナレッジベースへの接続に失敗しました。"
            st.error(error_text)
            logger.error(f"Vector database initialization error: {e}")
            st.stop()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'language' not in st.session_state:
            st.session_state.language = 'English'

    def setup_language_prompts(self):
        """Setup language-specific prompts and text"""
        self.language_config = {
            'English': {
                'title': 'Document Intelligence Assistant',
                'subtitle': 'Get instant answers from your organization\'s knowledge base',
                'input_placeholder': 'Ask me anything about your documents:',
                'ask_button': '🔍 Ask Question',
                'new_chat': '🔄 New Chat',
                'clear_history': '🗑️ Clear Chat History',
                'recent_queries': '📝 Recent Queries',
                'response_header': '### Response',
                'visual_resources': '### Related Visual Resources',
                'quick_examples': '### 💡 Quick Start Examples',
                'documents_analyzed': 'Documents Analyzed',
                'content_types': 'Content Types',
                'visual_resources_metric': 'Visual Resources',
                'source_documents': 'Source Documents',
                'searching': 'Searching knowledge base...',
                'analyzing': 'Analyzing documents and generating response...',
                'no_docs_found': 'No relevant documents found. Please try rephrasing your question.',
                'enter_question': 'Please enter a question to get started.',
                'processing_error': 'An error occurred while processing your request. Please try again.',
                'examples': [
                    ("📊 Data Management", "How to create and manage saved views?"),
                    ("🔄 Process Workflows", "What are the cycle count procedures?"),
                    ("🔍 Search & Filter", "How to use advanced filtering options?"),
                    ("👥 User Management", "How to manage user roles and permissions?"),
                    ("📋 Reporting", "How to generate comprehensive reports?"),
                    ("⚙️ Configuration", "How to configure system settings?")
                ]
            },
            'Japanese': {
                'title': 'ドキュメント インテリジェンス アシスタント',
                'subtitle': '組織のナレッジベースから即座に回答を取得',
                'input_placeholder': 'ドキュメントについて何でもお聞きください：',
                'ask_button': '🔍 質問する',
                'new_chat': '🔄 新しいチャット',
                'clear_history': '🗑️ 履歴をクリア',
                'recent_queries': '📝 最近の質問',
                'response_header': '### 回答',
                'visual_resources': '### 関連するビジュアルリソース',
                'quick_examples': '### 💡 クイックスタートの例',
                'documents_analyzed': '分析されたドキュメント',
                'content_types': 'コンテンツタイプ',
                'visual_resources_metric': 'ビジュアルリソース',
                'source_documents': 'ソースドキュメント',
                'searching': 'ナレッジベースを検索中...',
                'analyzing': 'ドキュメントを分析して回答を生成中...',
                'no_docs_found': '関連するドキュメントが見つかりませんでした。質問を言い換えてみてください。',
                'enter_question': '開始するには質問を入力してください。',
                'processing_error': 'リクエストの処理中にエラーが発生しました。再試行してください。',
                'examples': [
                    ("📊 データ管理", "保存ビューの作成と管理方法は？"),
                    ("🔄 プロセスワークフロー", "サイクルカウントの手順は何ですか？"),
                    ("🔍 検索とフィルター", "高度なフィルタリングオプションの使用方法は？"),
                    ("👥 ユーザー管理", "ユーザーの役割と権限の管理方法は？"),
                    ("📋 レポート", "包括的なレポートの生成方法は？"),
                    ("⚙️ 設定", "システム設定の構成方法は？")
                ]
            }
        }

    def get_text(self, key):
        """Get localized text based on current language"""
        return self.language_config[st.session_state.language][key]

    @staticmethod
    def convert_github_url_to_raw(github_url):
        """Convert GitHub URL to raw format"""
        if "github.com" in github_url and "/blob/" in github_url:
            return github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        return github_url

    @staticmethod
    def is_gif_file(url):
        """Check if URL points to a GIF file"""
        return url.lower().endswith('.gif')

    def display_image_safely(self, image_url, container):
        """Display image with error handling"""
        try:
            raw_url = self.convert_github_url_to_raw(image_url)
            
            with container:
                if self.is_gif_file(raw_url):
                    st.markdown(f"""
                    <div class="image-container">
                        <img src="{raw_url}" alt="Visual Resource" 
                             style="max-width: 100%; height: auto; border-radius: 4px;">
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    response = requests.get(raw_url, timeout=10)
                    response.raise_for_status()
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_container_width=True)
            return True
        except Exception as e:
            logger.error(f"Image display error: {e}")
            with container:
                error_text = "Unable to load image" if st.session_state.language == 'English' else "画像を読み込めません"
                st.error(error_text)
            return False

    def extract_images_from_documents(self, docs, max_images=3):
        """Extract images from retrieved documents"""
        all_images = []
        seen_images = set()
        
        for doc in docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                images_str = doc.metadata.get('images', '')
                
                if images_str and images_str.strip():
                    # Handle multiple delimiter formats
                    for delimiter in ['|', ',', ';']:
                        if delimiter in images_str:
                            images = [img.strip() for img in images_str.split(delimiter) if img.strip()]
                            break
                    else:
                        images = [images_str.strip()] if images_str.strip() else []
                    
                    for img in images:
                        if img and img not in seen_images and len(all_images) < max_images:
                            all_images.append(img)
                            seen_images.add(img)
        
        return all_images

    def generate_response(self, query, docs):
        """Generate AI response from documents"""
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = metadata.get('source', 'Document')
            doc_title = metadata.get('document_title', source)
            section = metadata.get('section', '')
            doc_type = metadata.get('document_type', 'Content')
            
            header = f"[Source {i+1}: {doc_title}"
            if section:
                header += f" - {section}"
            header += f" ({doc_type})]"
            
            context_parts.append(f"{header}\n{doc.page_content}")
        
        context = "\n\n---\n\n".join(context_parts)

        # Create language-specific prompt
        if st.session_state.language == 'Japanese':
            prompt = f"""
プロフェッショナルなドキュメントアシスタントとして、提供されたコンテキストに基づいて包括的で正確な回答を日本語で提供してください。

ガイドライン:
- 明確でプロフェッショナルな日本語を使用する
- 回答を論理的に構成し、適切な書式設定を行う
- 回答に「(ソース1、2、3)」のようなソース引用や参照を含めない
- 情報を直接的で権威ある知識として提示する
- 情報が不完全な場合は制限を認める
- 可能な場合は実用的なガイダンスを提供する
- ステップバイステップの指示には箇条書きや番号付きリストを使用する
- 会話的でありながらプロフェッショナルな回答を維持する

ナレッジベースからのコンテキスト:
{context}

ユーザーの質問: {query}

ソース引用なしで詳細なプロフェッショナルな回答を日本語で提供してください:
"""
        else:
            prompt = f"""
As a professional document assistant, provide a comprehensive and accurate answer based on the provided context.

Guidelines:
- Use clear, professional language
- Structure your response logically with proper formatting
- Do NOT include source citations or references like "(Source 1, 2, 3)" in your response
- Present information as direct, authoritative knowledge
- If information is incomplete, acknowledge limitations
- Provide actionable guidance when possible
- Use bullet points or numbered lists for step-by-step instructions
- Keep responses conversational yet professional

Context from knowledge base:
{context}

User Question: {query}

Please provide a detailed, professional response without any source citations:
"""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"AI response generation error: {e}")
            error_text = "I apologize, but I'm unable to generate a response at this time. Please try again or contact support." if st.session_state.language == 'English' else "申し訳ございませんが、現在回答を生成できません。再試行するかサポートにお問い合わせください。"
            return error_text

    def process_query(self, query, container=None):
        """Main query processing function"""
        try:
            # Use provided container or create a new one
            if container is None:
                container = st.container()
            
            with container:
                # Search documents - using default k value instead of hardcoded 50
                with st.spinner(self.get_text('searching')):
                    docs = self.vector_db.similarity_search(query, k=10)  # Changed from k=50 to k=10
                
                if not docs:
                    st.warning(self.get_text('no_docs_found'))
                    return
                
                # Generate response
                with st.spinner(self.get_text('analyzing')):
                    response_text = self.generate_response(query, docs)
                
                # Display response
                st.markdown('<div class="response-container">', unsafe_allow_html=True)
                st.markdown(self.get_text('response_header'))
                st.write(response_text)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display images - hardcoded to max 3 images
                images = self.extract_images_from_documents(docs, max_images=3)
                if images:
                    st.markdown(self.get_text('visual_resources'))
                    cols = st.columns(min(len(images), 3))
                    for idx, img_url in enumerate(images):
                        self.display_image_safely(img_url, cols[idx % len(cols)])
                
                # Add to chat history
                timestamp = datetime.now().strftime("%H:%M")
                st.session_state.chat_history.append({
                    'timestamp': timestamp,
                    'query': query,
                    'response': response_text,
                    'doc_count': len(docs),
                    'image_count': len(images)
                })
            
        except Exception as e:
            logger.error(f"Query processing error: {e}")
            if container:
                with container:
                    st.error(self.get_text('processing_error'))
            else:
                st.error(self.get_text('processing_error'))

    def render_sidebar(self):
        """Render sidebar with language selection and chat history"""
        with st.sidebar:
            # Language Selection at the top
            st.markdown("""
            <div class="language-selector">
                <h3>🌐 Language / 言語</h3>
            </div>
            """, unsafe_allow_html=True)
            
            language_options = {'English': '🇺🇸 English', 'Japanese': '🇯🇵 日本語'}
            
            new_language = st.selectbox(
                "",
                options=list(language_options.keys()),
                format_func=lambda x: language_options[x],
                index=0 if st.session_state.language == 'English' else 1,
                key="language_selector"
            )
            
            if new_language != st.session_state.language:
                st.session_state.language = new_language
                st.rerun()
            
            st.markdown("---")
            
            # Clear history button
            if st.button(self.get_text('clear_history')):
                st.session_state.chat_history = []
                st.rerun()
            
            # Chat history
            if st.session_state.chat_history:
                st.markdown(f"### {self.get_text('recent_queries')}")
                for idx, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                    with st.expander(f"{chat['timestamp']} - {chat['query'][:30]}..."):
                        query_label = "**Query:**" if st.session_state.language == 'English' else "**質問:**"
                        docs_label = "**Documents:**" if st.session_state.language == 'English' else "**ドキュメント:**"
                        images_label = "**Images:**" if st.session_state.language == 'English' else "**画像:**"
                        
                        st.write(f"{query_label} {chat['query']}")
                        st.write(f"{docs_label} {chat['doc_count']}")
                        st.write(f"{images_label} {chat['image_count']}")

    def render_main_interface(self):
        """Render main chat interface"""
        # Header
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem 0;">
            <h1 style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                       font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">
                {self.get_text('title')}
            </h1>
            <p style="color: #6c757d; font-size: 1.1rem;">
                {self.get_text('subtitle')}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Render sidebar
        self.render_sidebar()
        
        # Main query interface
        with st.container():
            query = st.text_input(
                "",
                placeholder=self.get_text('input_placeholder'),
                key="main_query"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                submit_clicked = st.button(self.get_text('ask_button'), type="primary", key="submit_main")
            with col2:
                if st.button(self.get_text('new_chat'), key="clear_main"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Response area container - positioned directly after input
        response_container = st.container()
        
        # Process query from input box
        if submit_clicked and query.strip():
            self.process_query(query, response_container)
        elif submit_clicked:
            with response_container:
                st.warning(self.get_text('enter_question'))
        
        # Quick examples
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        st.markdown(self.get_text('quick_examples'))
        
        examples = self.get_text('examples')
        
        # Check if any example button was clicked
        example_clicked = None
        cols = st.columns(2)
        for idx, (category, example) in enumerate(examples):
            with cols[idx % 2]:
                if st.button(f"{category}: {example}", key=f"example_{idx}"):
                    example_clicked = example
        
        # Process example query in the response container
        if example_clicked:
            self.process_query(example_clicked, response_container)

def main():
    """Main application entry point"""
    try:
        chatbot = ProfessionalChatbot()
        chatbot.render_main_interface()
    except Exception as e:
        error_text = "Failed to initialize the application. Please contact support." if st.session_state.get('language', 'English') == 'English' else "アプリケーションの初期化に失敗しました。サポートにお問い合わせください。"
        st.error(error_text)
        logger.error(f"Application initialization error: {e}")

if __name__ == "__main__":
    main()
