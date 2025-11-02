import streamlit as st
import requests
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG with Long-Term Memory",
    page_icon="brain",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.memory-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.source-card {
    background-color: #e8f4f8;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.3rem 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("RAG with Long-Term Memory")
st.markdown("Ask questions and the system will remember across sessions!")

# Sidebar
with st.sidebar:
    st.header("Settings")
    session_id = st.text_input("Session ID", value="default", help="Unique identifier for this conversation")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    
    # Memory stats
    st.subheader("Memory Statistics")
    try:
        stats_response = requests.get(f"{API_URL}/memory/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            st.metric("Total Memories", stats.get("total_memories", 0))
            st.metric("Episodic", stats.get("episodic_memories", 0))
            st.metric("Semantic", stats.get("semantic_memories", 0))
            st.metric("Avg Strength", f"{stats.get('average_strength', 0):.2f}")
        else:
            st.warning("Could not fetch memory stats")
    except Exception as e:
        st.error(f"API connection error: {str(e)}")
    
    st.divider()
    
    # Performance info
    st.subheader("⚡ Indexing Performance")
    st.info("💡 Tip: Check the terminal where you started the API to see detailed indexing timing information for each document uploaded.")
    if st.session_state.get("last_indexing_stats"):
        stats = st.session_state.last_indexing_stats
        st.success(f"Last indexing: {stats['time']:.3f}s for {stats['size']:,} chars")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Chat", "Documents", "Memories", "Add Fact"])

# Chat Tab
with tab1:
    st.header("Chat with RAG")
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Query input
    query = st.text_input("Ask a question:", key="query_input")
    
    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("Ask", type="primary")
    with col2:
        clear_button = st.button("Clear History")
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and query:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "query": query,
                        "session_id": session_id,
                        "top_k": top_k,
                        "temperature": temperature,
                        "include_sources": True
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chat_history.append({
                        "query": query,
                        "answer": data["answer"],
                        "sources": data["sources"],
                        "memories": data["memories_used"],
                        "metadata": data["metadata"],
                        "timestamp": datetime.now()
                    })
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.divider()
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['answer']}")
                
                # Show sources
                if chat.get('sources'):
                    with st.expander(f"Sources ({len(chat['sources'])})"):
                        for source in chat['sources']:
                            st.markdown(f"**[{source['id']}]** (score: {source['score']:.3f})")
                            st.text(source['content'][:200] + "..." if len(source['content']) > 200 else source['content'])
                
                # Show memories used
                if chat.get('memories'):
                    with st.expander(f"Memories Used ({len(chat['memories'])})"):
                        for mem in chat['memories']:
                            st.markdown(f"**Type:** {mem['type']} | **Importance:** {mem['importance']:.2f}")
                            st.text(mem['content'][:200] + "..." if len(mem['content']) > 200 else mem['content'])
                
                st.caption(f"Time: {chat['timestamp'].strftime('%H:%M:%S')}")
                st.divider()

# Documents Tab
with tab2:
    st.header("Document Management")
    
    # Display supported file types
    try:
        info_response = requests.get(f"{API_URL}/")
        if info_response.status_code == 200:
            info_data = info_response.json()
            supported_types = info_data.get("supported_file_types", [".pdf", ".txt", ".md"])
            st.info(f"Supported file types: {', '.join(supported_types)}")
    except Exception:
        st.info("Supported file types: .pdf, .txt, .md")
    
    # File upload
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'md', 'pdf', 'text', 'markdown'])
    
    if uploaded_file and st.button("Upload & Index"):
        with st.spinner("Uploading and indexing..."):
            start_time = datetime.now()
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/documents/upload", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()
                    
                    st.success(f"Document '{uploaded_file.name}' uploaded successfully!")
                    
                    # Show detailed timing information
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("File Size", f"{result.get('size', 0):,} chars")
                    with col2:
                        st.metric("Processing Time", f"{result.get('processing_time', 0):.3f}s")
                    with col3:
                        st.metric("Indexing Time", f"{result.get('indexing_time', 0):.3f}s")
                    with col4:
                        st.metric("Total Time", f"{result.get('total_time', duration):.3f}s")
                    
                    st.info(f"File type: {result.get('file_type', 'unknown')}")
                    
                    # Store stats for sidebar display
                    st.session_state.last_indexing_stats = {
                        "time": result.get('total_time', duration),
                        "size": result.get('size', 0)
                    }
                elif response.status_code == 400:
                    error_data = response.json()
                    st.error(f"Error: {error_data.get('detail', 'Bad request')}")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Text input
    st.subheader("Add Text Directly")
    doc_text = st.text_area("Enter text to index:", height=200)
    doc_id = st.text_input("Document ID:", value=f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    if st.button("Add Text") and doc_text:
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
                    
                    st.success(f"Text added with ID: {doc_id}")
                    
                    # Show timing information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Text Size", f"{result.get('size', len(doc_text)):,} chars")
                    with col2:
                        st.metric("Indexing Time", f"{result.get('indexing_time', duration):.3f}s")
                    
                    # Store stats for sidebar display
                    st.session_state.last_indexing_stats = {
                        "time": result.get('indexing_time', duration),
                        "size": result.get('size', len(doc_text))
                    }
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Memories Tab
with tab3:
    st.header("Memory Search")
    
    search_query = st.text_input("Search memories:", key="memory_search")
    memory_type_filter = st.selectbox("Filter by type:", ["All", "episodic", "semantic"])
    search_k = st.slider("Number of memories:", min_value=1, max_value=20, value=10)
    
    if st.button("Search Memories") and search_query:
        with st.spinner("Searching..."):
            try:
                params = {"query": search_query, "k": search_k}
                if memory_type_filter != "All":
                    params["memory_type"] = memory_type_filter
                
                response = requests.get(f"{API_URL}/memory/search", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Found {data['count']} memories")
                    
                    for mem in data['memories']:
                        with st.container():
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**Type:** {mem['type']}")
                            with col2:
                                st.markdown(f"**Importance:** {mem['importance']:.2f}")
                            with col3:
                                st.markdown(f"**Strength:** {mem['strength']:.2f}")
                            
                            st.text(mem['content'])
                            st.caption(f"Created: {mem['created_at'][:19]} | Accessed: {mem['access_count']} times")
                            st.divider()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Add Fact Tab
with tab4:
    st.header("Add Fact to Memory")
    st.markdown("Manually add important facts to the long-term memory system.")
    
    fact_text = st.text_area("Fact to remember:", height=150, 
                             placeholder="Enter an important fact, preference, or piece of knowledge...")
    
    col1, col2 = st.columns(2)
    with col1:
        fact_importance = st.slider("Importance:", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    with col2:
        fact_tags = st.text_input("Tags (comma-separated):", placeholder="tag1, tag2, tag3")
    
    if st.button("Add Fact", type="primary") and fact_text:
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
                    st.success("Fact added to long-term memory!")
                    st.balloons()
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
st.markdown("**RAG with Long-Term Memory** | Powered by FastAPI & Streamlit")
