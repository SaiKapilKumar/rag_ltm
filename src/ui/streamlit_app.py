import streamlit as st
import requests
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG with Long-Term Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - ChatGPT-like styling
st.markdown("""
<style>
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main > div:first-child > header {visibility: hidden;}
    
    /* Main container */
    .main {
        padding-top: 1rem;
        padding-bottom: 6rem;
        max-width: 100%;
    }
    
    /* Ensure proper spacing for chat */
    .main .block-container {
        padding-bottom: 6rem;
    }
    
    /* Chat messages container */
    .stChatMessage {
        padding: 1.5rem 1rem;
        margin-bottom: 1rem;
        max-width: 100%;
    }
    
    /* User message - aligned to the right */
    .stChatMessage[data-testid="user-message"] {
        background-color: #2d2d2d !important;
        margin-left: auto;
        margin-right: 0;
        max-width: 70%;
        border-radius: 18px;
        display: flex;
        flex-direction: row-reverse;
    }
    
    .stChatMessage[data-testid="user-message"] .stMarkdown {
        text-align: left;
        color: #ececec;
    }
    
    /* Assistant message - centered/left aligned */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: transparent !important;
        margin-left: 0;
        margin-right: auto;
        max-width: 100%;
        border-radius: 0;
    }
    
    .stChatMessage[data-testid="assistant-message"] .stMarkdown {
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Avatar styling */
    .stChatMessage [data-testid="chatAvatarIcon-user"] {
        background-color: #19c37d;
    }
    
    .stChatMessage [data-testid="chatAvatarIcon-assistant"] {
        background-color: transparent;
    }
    
    /* Chat input */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background-color: #343541;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 100;
    }
    
    .stChatInput > div {
        max-width: 48rem;
        margin: 0 auto;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #202123 !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        background-color: #202123 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stButton button,
    section[data-testid="stSidebar"] label {
        color: #ececec !important;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        border: 1px solid #565869;
        padding: 0.5rem 1rem;
        font-weight: 500;
        background-color: #40414f;
        color: #ececec;
    }
    
    .stButton > button:hover {
        background-color: #4d4d5c;
        border-color: #7a7a8c;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.875rem;
        color: #9b9b9b;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    .streamlit-expanderContent {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #ececec;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #19c37d;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        color: #ececec !important;
    }
    
    /* Input fields */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        background-color: #40414f !important;
        color: #ececec !important;
        border: 1px solid #565869 !important;
        border-radius: 8px;
    }
    
    /* Slider */
    .stSlider {
        color: #ececec;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.05);
        color: #ececec;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # New Chat button at the top
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Session settings
    st.subheader("👤 User & Session")
    user_id = st.text_input("User ID", value="user1", help="Your unique user identifier (name, number, etc.)")
    session_id = st.text_input("Session ID", value="default", help="Unique identifier for this conversation")
    
    # Model parameters
    st.subheader("Model Parameters")
    top_k = st.slider("Results to retrieve", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    
    # Vector Store Management
    st.subheader("📚 Vector Store")
    try:
        doc_stats_response = requests.get(f"{API_URL}/documents/stats")
        if doc_stats_response.status_code == 200:
            doc_stats = doc_stats_response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", doc_stats.get("total_documents", 0))
            with col2:
                st.metric("Chunks", doc_stats.get("total_chunks", 0))
            
            # Clear vector store button
            if st.button("🗑️ Clear Vector Store", use_container_width=True, type="secondary"):
                try:
                    clear_response = requests.delete(f"{API_URL}/documents/clear")
                    if clear_response.status_code == 200:
                        st.success("✅ Vector store cleared!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear vector store")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Could not fetch vector store stats")
    except Exception:
        st.error("Connection error")
    
    st.divider()
    
    # Memory stats
    st.subheader("📊 Memory Statistics")
    try:
        stats_response = requests.get(f"{API_URL}/memory/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", stats.get("total_memories", 0))
                st.metric("Episodic", stats.get("episodic_memories", 0))
            with col2:
                st.metric("Semantic", stats.get("semantic_memories", 0))
                st.metric("Avg Strength", f"{stats.get('average_strength', 0):.2f}")
            
            # Clear all memories button
            if st.button("🗑️ Clear All Memories", use_container_width=True, type="secondary"):
                try:
                    clear_response = requests.delete(f"{API_URL}/memory/clear")
                    if clear_response.status_code == 200:
                        st.success("✅ All memories cleared!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to clear memories")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Could not fetch memory stats")
    except Exception:
        st.error("Connection error")
    
    st.divider()
    
    # Performance info
    if st.session_state.get("last_indexing_stats"):
        st.subheader("⚡ Last Indexing")
        stats = st.session_state.last_indexing_stats
        st.success(f"{stats['time']:.2f}s for {stats['size']:,} chars")

# Main content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat", "📄 Documents", "🧠 Memories", "➕ Add Fact", "📊 Dashboard"])

# Chat Tab
with tab1:
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message.get("sources"):
                with st.expander(f"📚 View {len(message['sources'])} sources"):
                    for idx, source in enumerate(message['sources'], 1):
                        st.markdown(f"**Source {idx}** (relevance: {source['score']:.2%})")
                        st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                        if idx < len(message['sources']):
                            st.divider()
            
            # Show memories if available
            if message.get("memories"):
                with st.expander(f"🧠 Used {len(message['memories'])} memories"):
                    for idx, mem in enumerate(message['memories'], 1):
                        st.markdown(f"**Memory {idx}** • {mem['type']} • Importance: {mem['importance']:.2f}")
                        st.text(mem['content'][:300] + "..." if len(mem['content']) > 300 else mem['content'])
                        if idx < len(message['memories']):
                            st.divider()
    
    # Accept user input
    if prompt := st.chat_input("Message"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={
                            "query": prompt,
                            "session_id": session_id,
                            "user_id": user_id,
                            "top_k": top_k,
                            "temperature": temperature,
                            "include_sources": True
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])
                        memories = data.get("memories_used", [])
                        
                        # Display the answer
                        st.markdown(answer)
                        
                        # Show sources
                        if sources:
                            with st.expander(f"📚 View {len(sources)} sources"):
                                for idx, source in enumerate(sources, 1):
                                    st.markdown(f"**Source {idx}** (relevance: {source['score']:.2%})")
                                    st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                                    if idx < len(sources):
                                        st.divider()
                        
                        # Show memories
                        if memories:
                            with st.expander(f"🧠 Used {len(memories)} memories"):
                                for idx, mem in enumerate(memories, 1):
                                    st.markdown(f"**Memory {idx}** • {mem['type']} • Importance: {mem['importance']:.2f}")
                                    st.text(mem['content'][:300] + "..." if len(mem['content']) > 300 else mem['content'])
                                    if idx < len(memories):
                                        st.divider()
                        
                        # Add assistant response to chat history
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources,
                            "memories": memories
                        })
                    else:
                        error_msg = f"❌ Error: {response.status_code}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Documents Tab
with tab2:
    st.markdown("### 📄 Document Management")
    st.caption("Upload and index documents for the RAG system")
    st.divider()
    
    # Display currently indexed documents
    st.subheader("📚 Indexed Documents")
    try:
        docs_response = requests.get(f"{API_URL}/documents/list")
        if docs_response.status_code == 200:
            docs_data = docs_response.json()
            if docs_data['count'] > 0:
                st.success(f"✅ {docs_data['count']} document(s) currently indexed")
                
                # Display documents in a table-like format
                for idx, doc in enumerate(docs_data['documents'], 1):
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            st.markdown(f"**{idx}. {doc['filename']}**")
                        with col2:
                            st.caption(f"Chunks: {doc['chunk_count']}")
                        with col3:
                            st.caption(f"Source: {doc['source']}")
                        
                        if idx < docs_data['count']:
                            st.divider()
            else:
                st.info("📝 No documents indexed yet. Upload a document below to get started.")
        else:
            st.warning("Could not fetch document list")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Display supported file types
    try:
        info_response = requests.get(f"{API_URL}/")
        if info_response.status_code == 200:
            info_data = info_response.json()
            supported_types = info_data.get("supported_file_types", [".pdf", ".txt", ".md"])
            st.info(f"📎 Supported file types: {', '.join(supported_types)}")
    except Exception:
        st.info("📎 Supported file types: .pdf, .txt, .md")
    
    # File upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md', 'pdf', 'text', 'markdown'], 
                                     label_visibility="collapsed")
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(f"📄 {uploaded_file.name}")
        with col2:
            upload_btn = st.button("Upload & Index", type="primary", use_container_width=True)
        
        if upload_btn:
            with st.spinner("Processing document..."):
                start_time = datetime.now()
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/documents/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        end_time = datetime.now()
                        duration = (end_time - start_time).total_seconds()
                        
                        st.success(f"✅ Document '{uploaded_file.name}' uploaded successfully!")
                        
                        # Show timing information
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("File Size", f"{result.get('size', 0):,} chars")
                        with col2:
                            st.metric("Processing", f"{result.get('processing_time', 0):.2f}s")
                        with col3:
                            st.metric("Indexing", f"{result.get('indexing_time', 0):.2f}s")
                        
                        # Store stats for sidebar display
                        st.session_state.last_indexing_stats = {
                            "time": result.get('total_time', duration),
                            "size": result.get('size', 0)
                        }
                    elif response.status_code == 400:
                        error_data = response.json()
                        st.error(f"❌ Error: {error_data.get('detail', 'Bad request')}")
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    st.divider()
    
    # Text input
    st.subheader("Add Text Directly")
    doc_text = st.text_area("Enter text to index:", height=200, 
                           placeholder="Paste or type text here...",
                           label_visibility="collapsed")
    doc_id = st.text_input("Document ID (optional):", 
                          value=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                          label_visibility="collapsed")
    
    if st.button("Add Text", type="primary", disabled=not doc_text) and doc_text:
        with st.spinner("Indexing text..."):
            start_time = datetime.now()
            try:
                response = requests.post(
                    f"{API_URL}/documents/add",
                    json={"text": doc_text, "doc_id": doc_id, "metadata": {"source": "direct_input"}}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    st.success(f"✅ Text indexed successfully (ID: {doc_id})")
                    
                    # Show timing information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Text Size", f"{result.get('size', len(doc_text)):,} chars")
                    with col2:
                        st.metric("Indexing Time", f"{result.get('indexing_time', duration):.2f}s")
                    
                    # Store stats for sidebar display
                    st.session_state.last_indexing_stats = {
                        "time": result.get('indexing_time', duration),
                        "size": result.get('size', len(doc_text))
                    }
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Memories Tab
with tab3:
    st.markdown("### 🧠 Memory Search")
    st.caption("Search and explore the long-term memory system")
    st.divider()
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search memories:", key="memory_search", 
                                    placeholder="Enter search query...",
                                    label_visibility="collapsed")
    with col2:
        memory_type_filter = st.selectbox("Type:", ["All", "episodic", "semantic"],
                                         label_visibility="collapsed")
    
    search_k = st.slider("Number of results:", min_value=1, max_value=20, value=10)
    
    if st.button("Search", type="primary", disabled=not search_query) and search_query:
        with st.spinner("Searching memories..."):
            try:
                params = {"query": search_query, "k": search_k}
                if memory_type_filter != "All":
                    params["memory_type"] = memory_type_filter
                
                response = requests.get(f"{API_URL}/memory/search", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Found {data['count']} memories")
                    
                    if data['count'] > 0:
                        st.divider()
                        for idx, mem in enumerate(data['memories'], 1):
                            with st.container():
                                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                                with col1:
                                    st.markdown(f"**Memory {idx}**")
                                with col2:
                                    st.caption(f"Type: {mem['type']}")
                                with col3:
                                    st.caption(f"Importance: {mem['importance']:.2f}")
                                with col4:
                                    st.caption(f"Strength: {mem['strength']:.2f}")
                                
                                st.markdown(mem['content'])
                                st.caption(f"🕐 Created: {mem['created_at'][:19]} • Accessed: {mem['access_count']} times")
                                
                                if idx < data['count']:
                                    st.divider()
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Add Fact Tab
with tab4:
    st.markdown("### ➕ Add Fact to Memory")
    st.caption("Manually store important facts in the long-term memory system")
    st.divider()
    
    fact_text = st.text_area("Fact to remember:", height=150, 
                             placeholder="Enter an important fact, preference, or piece of knowledge...",
                             label_visibility="collapsed")
    
    col1, col2 = st.columns(2)
    with col1:
        fact_importance = st.slider("Importance:", min_value=0.0, max_value=1.0, value=0.8, step=0.1,
                                   help="How important is this fact? (0.0 = low, 1.0 = high)")
    with col2:
        fact_tags = st.text_input("Tags (comma-separated):", placeholder="tag1, tag2, tag3",
                                 label_visibility="collapsed")
    
    if st.button("💾 Save to Memory", type="primary", disabled=not fact_text, use_container_width=True) and fact_text:
        with st.spinner("Storing fact..."):
            try:
                tags_list = [tag.strip() for tag in fact_tags.split(",")] if fact_tags else []
                
                response = requests.post(
                    f"{API_URL}/memory/fact",
                    json={
                        "fact": fact_text,
                        "importance": fact_importance,
                        "tags": tags_list
                    }
                )
                
                if response.status_code == 200:
                    st.success("✅ Fact successfully added to long-term memory!")
                    st.balloons()
                    # Clear the form
                    st.session_state.fact_text = ""
                else:
                    st.error(f"❌ Error: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Dashboard Tab
with tab5:
    st.markdown("### 📊 Memory Dashboard")
    st.caption("View memory usage, frequent topics, and conversation history")
    st.divider()
    
    # User/Session Selector
    col1, col2 = st.columns(2)
    with col1:
        view_type = st.radio("View by:", ["Current User", "All Users", "Specific Session"], horizontal=True)
    
    selected_user = None
    selected_session = None
    
    if view_type == "Current User":
        selected_user = user_id
        st.info(f"👤 Viewing data for user: **{user_id}**")
    elif view_type == "Specific Session":
        # Fetch available sessions
        try:
            sessions_response = requests.get(f"{API_URL}/sessions")
            if sessions_response.status_code == 200:
                sessions_data = sessions_response.json()
                if sessions_data['count'] > 0:
                    session_options = [f"{s['session_id']} (User: {s['user_id']}, Last active: {s['last_active'][:19]})" 
                                     for s in sessions_data['sessions']]
                    selected_idx = st.selectbox("Select session:", range(len(session_options)), 
                                                format_func=lambda x: session_options[x])
                    selected_session = sessions_data['sessions'][selected_idx]['session_id']
                    st.info(f"📝 Viewing data for session: **{selected_session}**")
                else:
                    st.warning("No sessions found")
            else:
                st.error("Could not fetch sessions")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    if view_type != "Specific Session" or selected_session:
        st.divider()
        
        # Memory Statistics
        st.subheader("📈 Memory Statistics")
        try:
            if selected_session:
                stats_response = requests.get(f"{API_URL}/sessions/{selected_session}/stats")
            elif selected_user:
                stats_response = requests.get(f"{API_URL}/users/{selected_user}/stats")
            else:
                stats_response = requests.get(f"{API_URL}/memory/stats")
            
            if stats_response.status_code == 200:
                stats = stats_response.json()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Memories", stats.get("total_memories", 0))
                with col2:
                    st.metric("Episodic", stats.get("episodic_memories", stats.get("episodic_count", 0)))
                with col3:
                    st.metric("Semantic", stats.get("semantic_memories", stats.get("semantic_count", 0)))
                with col4:
                    if "total_storage_bytes" in stats:
                        storage_mb = stats["total_storage_bytes"] / (1024 * 1024)
                        st.metric("Storage", f"{storage_mb:.2f} MB")
                    else:
                        st.metric("Avg Strength", f"{stats.get('average_strength', 0):.2f}")
                
                # Additional user stats
                if "total_sessions" in stats:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sessions", stats.get("total_sessions", 0))
                    with col2:
                        st.metric("Avg/Session", f"{stats.get('avg_memories_per_session', 0):.1f}")
                    with col3:
                        st.metric("Avg Importance", f"{stats.get('avg_importance', 0):.2f}")
            else:
                st.warning("Could not fetch statistics")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Frequent Topics
        st.subheader("🔥 Frequently Discussed Topics")
        col1, col2 = st.columns([3, 1])
        with col2:
            topics_count = st.slider("Show top:", min_value=5, max_value=20, value=10, key="topics_slider")
        
        try:
            if selected_session:
                topics_response = requests.get(f"{API_URL}/sessions/{selected_session}/frequent-topics",
                                              params={"top_n": topics_count})
            elif selected_user:
                topics_response = requests.get(f"{API_URL}/users/{selected_user}/frequent-topics",
                                              params={"top_n": topics_count})
            else:
                # Get all users and show combined topics
                st.info("Select a user or session to view frequent topics")
                topics_response = None
            
            if topics_response and topics_response.status_code == 200:
                topics_data = topics_response.json()
                if topics_data['count'] > 0:
                    # Create a bar chart using Streamlit's native chart
                    import pandas as pd
                    topics_df = pd.DataFrame(topics_data['topics'])
                    
                    # Display as bar chart
                    st.bar_chart(topics_df.set_index('topic')['count'])
                    
                    # Also show as table
                    with st.expander("📋 View detailed topic data"):
                        st.dataframe(topics_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No topics found. Start asking questions to build up memory!")
            elif topics_response:
                st.warning("Could not fetch topics")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
        
        st.divider()
        
        # Query Patterns
        if selected_session or selected_user:
            st.subheader("📊 Query Patterns")
            try:
                if selected_session:
                    patterns_response = requests.get(f"{API_URL}/sessions/{selected_session}/patterns")
                elif selected_user:
                    patterns_response = requests.get(f"{API_URL}/users/{selected_user}/patterns")
                
                if patterns_response.status_code == 200:
                    patterns = patterns_response.json()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Queries", patterns.get("total_queries", 0))
                        st.metric("Avg Query Length", f"{patterns.get('avg_query_length', 0):.1f} words")
                    
                    with col2:
                        # Question types
                        if patterns.get("question_types"):
                            st.write("**Question Types:**")
                            for q_type, count in sorted(patterns["question_types"].items(), 
                                                       key=lambda x: x[1], reverse=True):
                                st.text(f"  {q_type}: {count}")
                else:
                    st.warning("Could not fetch query patterns")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            
            st.divider()
        
        # Conversation Timeline
        if selected_session:
            st.subheader("📜 Conversation Timeline")
            timeline_limit = st.slider("Show last N conversations:", min_value=5, max_value=50, value=20, key="timeline_slider")
            
            try:
                timeline_response = requests.get(f"{API_URL}/sessions/{selected_session}/timeline",
                                                params={"limit": timeline_limit})
                
                if timeline_response.status_code == 200:
                    timeline_data = timeline_response.json()
                    if timeline_data['count'] > 0:
                        for idx, item in enumerate(timeline_data['timeline'], 1):
                            with st.expander(f"💬 {item['timestamp'][:19]} - {item['query'][:50]}..."):
                                st.markdown(f"**Query:** {item['query']}")
                                st.markdown(f"**Answer:** {item['answer']}")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.caption(f"Importance: {item['importance']:.2f}")
                                with col2:
                                    st.caption(f"Accessed: {item['access_count']} times")
                    else:
                        st.info("No conversation history yet")
                else:
                    st.warning("Could not fetch timeline")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            
            st.divider()
        
        # Memory Management Actions
        st.subheader("🗑️ Memory Management")
        st.warning("⚠️ These actions cannot be undone!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if selected_session:
                st.write("**Clear Session Memories**")
                clear_type = st.radio("Clear:", ["Conversations only (episodic)", "All memories"], key="clear_type")
                
                if st.button("🗑️ Clear Session Memories", type="secondary", use_container_width=True):
                    try:
                        memory_type = "episodic" if "episodic" in clear_type else None
                        params = {"memory_type": memory_type} if memory_type else {}
                        clear_response = requests.delete(f"{API_URL}/sessions/{selected_session}/memories", 
                                                        params=params)
                        
                        if clear_response.status_code == 200:
                            result = clear_response.json()
                            st.success(f"✅ Cleared {result['deleted_count']} memories from session")
                            st.rerun()
                        else:
                            st.error("Failed to clear memories")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if selected_user:
                st.write("**Clear User Memories**")
                st.caption("This will clear ALL memories across all sessions")
                
                if st.button("🗑️ Clear All User Memories", type="secondary", use_container_width=True):
                    try:
                        clear_response = requests.delete(f"{API_URL}/users/{selected_user}/memories")
                        
                        if clear_response.status_code == 200:
                            result = clear_response.json()
                            st.success(f"✅ Cleared {result['deleted_count']} memories for user")
                            st.rerun()
                        else:
                            st.error("Failed to clear memories")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p>RAG with Long-Term Memory • Powered by FastAPI, Streamlit & Azure OpenAI</p>
</div>
""", unsafe_allow_html=True)
