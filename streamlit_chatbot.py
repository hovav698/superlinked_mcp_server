"""
Streamlit RAG Chatbot with REAL-TIME streaming and conversation memory.
Maintains a persistent Claude client for conversation continuity.
"""
import streamlit as st
import asyncio
import json
from dotenv import load_dotenv
import nest_asyncio
from claude_client import create_client, connect_client, disconnect_client, stream_query, get_model

# Allow nested event loops (fixes Streamlit async issues)
nest_asyncio.apply()

# Load environment variables
load_dotenv()


def get_or_create_client():
    """
    Get existing Claude client or create a new one.
    Manages client and event loop in session state.

    Returns:
        Tuple of (client, event_loop)
    """
    if 'client_loop' not in st.session_state:
        # Create a dedicated event loop for the client
        st.session_state.client_loop = asyncio.new_event_loop()

    if 'client' not in st.session_state or st.session_state.client is None:
        # Create and connect client
        loop = st.session_state.client_loop
        asyncio.set_event_loop(loop)

        client = create_client()

        # Connect the client
        loop.run_until_complete(connect_client(client))

        st.session_state.client = client
        st.session_state.client_connected = True

    return st.session_state.client, st.session_state.client_loop


def reset_client():
    """Reset the Claude client connection."""
    if 'client' in st.session_state and st.session_state.client is not None:
        try:
            loop = st.session_state.client_loop
            asyncio.set_event_loop(loop)
            loop.run_until_complete(disconnect_client(st.session_state.client))
        except:
            pass

    st.session_state.client = None
    st.session_state.client_connected = False


async def stream_query_with_updates(client, user_input: str, steps_placeholder):
    """
    Stream query and parse messages into steps with real-time updates.

    Args:
        client: ClaudeSDKClient instance
        user_input: User's query text
        steps_placeholder: Streamlit placeholder for displaying steps

    Returns:
        Tuple of (steps, final_result)
    """
    steps = []
    final_result = None

    async for message in stream_query(client, user_input):
        msg_type = type(message).__name__

        # Parse AssistantMessage blocks
        if msg_type == "AssistantMessage" and hasattr(message, 'content'):
            content = message.content

            if isinstance(content, list):
                for block in content:
                    block_type = type(block).__name__

                    # TextBlock - thinking
                    if block_type == "TextBlock" and hasattr(block, 'text'):
                        steps.append({
                            "type": "thinking",
                            "content": block.text
                        })
                        # Update display immediately
                        update_steps_display(steps_placeholder, steps)

                    # ToolUseBlock - tool call
                    elif block_type == "ToolUseBlock":
                        steps.append({
                            "type": "tool",
                            "name": block.name,
                            "input": block.input
                        })
                        # Update display immediately
                        update_steps_display(steps_placeholder, steps)

        # Parse UserMessage for tool results
        elif msg_type == "UserMessage" and hasattr(message, 'content'):
            content = message.content

            if isinstance(content, list):
                for block in content:
                    block_type = type(block).__name__

                    if block_type == "ToolResultBlock":
                        error = getattr(block, 'is_error', False)
                        result_content = getattr(block, 'content', 'No content')
                        steps.append({
                            "type": "tool_result",
                            "error": error,
                            "content": result_content
                        })
                        # Update display immediately
                        update_steps_display(steps_placeholder, steps)

        # Final result
        if msg_type == "ResultMessage" and hasattr(message, 'result'):
            final_result = message.result

    return steps, final_result


def update_steps_display(placeholder, steps):
    """
    Update the steps display in real-time.
    """
    with placeholder.container():
        st.markdown("---")
        st.markdown(f"**🔍 Execution Steps:** ({len(steps)} steps)")

        for i, step in enumerate(steps, 1):
            if step["type"] == "thinking":
                with st.expander(f"💭 Step {i}: Thinking", expanded=False):
                    st.info(step["content"])

            elif step["type"] == "tool":
                with st.expander(f"🔧 Step {i}: Using Tool - `{step['name']}`", expanded=True):
                    st.json(step["input"])

            elif step["type"] == "tool_result":
                with st.expander(f"✅ Step {i}: Tool Result", expanded=False):
                    if step["error"]:
                        st.error(f"Error: {step['content']}")
                    else:
                        # Try to parse and display as JSON if possible
                        try:
                            json_content = json.loads(step["content"])
                            st.json(json_content)
                        except:
                            st.success(step["content"])


def display_steps(steps):
    """
    Display intermediate steps (for chat history).
    """
    if not steps:
        return

    st.markdown("---")
    st.markdown(f"**🔍 Execution Steps:** ({len(steps)} steps)")

    for i, step in enumerate(steps, 1):
        if step["type"] == "thinking":
            with st.expander(f"💭 Step {i}: Thinking", expanded=False):
                st.info(step["content"])

        elif step["type"] == "tool":
            with st.expander(f"🔧 Step {i}: Using Tool - `{step['name']}`", expanded=False):
                st.json(step["input"])

        elif step["type"] == "tool_result":
            with st.expander(f"✅ Step {i}: Tool Result", expanded=False):
                if step["error"]:
                    st.error(f"Error: {step['content']}")
                else:
                    # Try to parse and display as JSON if possible
                    try:
                        json_content = json.loads(step["content"])
                        st.json(json_content)
                    except:
                        st.success(step["content"])

    st.markdown("---")


def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 RAG Chatbot with Superlinked")
    st.caption(f"Ask questions in natural language 💬⚡ | Model: {get_model()}")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "client" not in st.session_state:
        st.session_state.client = None
        st.session_state.client_connected = False

    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                # Show intermediate steps
                if "steps" in msg and msg["steps"]:
                    display_steps(msg["steps"])

                # Show final answer
                if msg["content"]:
                    st.markdown("**📄 Answer:**")
                    st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process query
        with st.chat_message("assistant"):
            # Create placeholder for real-time updates
            steps_placeholder = st.empty()

            # Show processing status
            status = st.status("🤔 Processing...", expanded=True)

            try:
                # Get or create persistent client
                client, loop = get_or_create_client()

                # Run query with persistent client
                asyncio.set_event_loop(loop)
                steps, final_result = loop.run_until_complete(
                    stream_query_with_updates(client, prompt, steps_placeholder)
                )

                # Update status
                status.update(label="✅ Complete!", state="complete", expanded=False)

                # Clear placeholder and show final organized view
                steps_placeholder.empty()

                # Display final steps
                if steps:
                    display_steps(steps)

                # Display final result
                if final_result:
                    st.markdown("**📄 Answer:**")
                    st.markdown(final_result)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_result,
                        "steps": steps
                    })
                else:
                    st.warning("No response received")

            except Exception as e:
                status.update(label="❌ Error", state="error", expanded=True)
                st.error(f"Error: {str(e)}")
                # Reset client on error
                reset_client()
                st.info("Client reset. Try your message again.")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **Claude AI** via the Agent SDK
        - **Superlinked** for vector search
        - **MCP** for tool integration

        ### Example Conversation:
        1. "I have this file: sample_data/business_news.csv, Who wanted to have strike?"
        2. "This news is old, put more weight on the date"
        3. "Any news related to gas? What does it say?"

        ### Features:
        - ⚡ **Real-time streaming** - See steps as they happen
        - 💬 **Conversation memory** - Multi-turn context
        - 👁️ All intermediate steps shown
        - 🔧 Tool usage visualization
        """)

        # Connection status
        if st.session_state.get('client_connected', False):
            st.success("🟢 Connected - Conversation active")
        else:
            st.info("⚪ Not connected - Send a message to start")

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat & Reset"):
            st.session_state.messages = []
            reset_client()
            st.rerun()

        st.divider()

        # Stats
        st.caption("**Session Stats:**")
        st.caption(f"📊 Total messages: {len(st.session_state.messages)}")
        user_msgs = sum(1 for m in st.session_state.messages if m['role'] == 'user')
        assistant_msgs = sum(1 for m in st.session_state.messages if m['role'] == 'assistant')
        st.caption(f"👤 User: {user_msgs} | 🤖 Assistant: {assistant_msgs}")


if __name__ == "__main__":
    main()
