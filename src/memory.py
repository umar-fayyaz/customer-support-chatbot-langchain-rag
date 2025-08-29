from langchain.memory import ConversationBufferMemory

_global_memory = None

def get_memory():
    global _global_memory
    if _global_memory is None:
        _global_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    return _global_memory
