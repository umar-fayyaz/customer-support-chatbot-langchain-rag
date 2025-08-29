import asyncio
from src import config
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from workflow.rag import load_pdf_file, text_split, build_rag_pipeline
from src.memory import get_memory  # <-- import shared memory

## Rag set up
from functools import lru_cache

@lru_cache
def rag_chain():
    extracted_data = load_pdf_file("Data/")
    text_chunks = text_split(extracted_data)
    return build_rag_pipeline(text_chunks)


class NewCustomerWorkflow:
    def __init__(self):
        # Shared memory
        self.memory = get_memory()

        # LLM
        self.low_llm = ChatOpenAI(
            model=config.LOW_LLM,
            temperature=0.7,
            openai_api_key=config.OPENAI_API_KEY
        )

        # ---------------- Onboarding Chain ---------------- #
        onboarding_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are the RemoteLock Assistant onboarding a NEW customer.

Conversation flow:
1. Greet politely and briefly ask how you can help related to smart locks.
2. If the user's need is clear, move directly to asking if they own any smart lock hardware.
3. If they say YES, ask for the brand.
4. If they say NO, ask for business type (vacation rental, small office, etc.).
5. Ask how many doors they plan to manage.
6. If >= 30 doors, recommend speaking to a sales rep.
7. If user agrees to speak to sales rep, give this link: https://calendly.com/
8. Ask for their name naturally during the conversation if not already known.

RULES:
- Avoid starting with "What's your name?" — start with a polite greeting and a relevant question about smart locks.
- Never re-ask a question already answered in chat history.
- If user agrees to meet sales rep (yes, sure, okay, let's do it, etc.), immediately give the link without asking again.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.onboarding_chain = (
            {
                "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"],
                "input": lambda x: x["input"]
            }
            | onboarding_prompt
            | self.low_llm
            | StrOutputParser()
        )

        # ---------------- Intent Chain ---------------- #
        intent_prompt = ChatPromptTemplate.from_messages([
            ("system", """
Classify the user input into one of:
- onboarding_flow → If user is answering/responding to onboarding questions or continuing setup.
- rag_query → If user is asking a general product question (outside onboarding script).
Output only one label.
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.intent_chain = (
            {
                "chat_history": lambda _: self.memory.load_memory_variables({})["chat_history"],
                "input": lambda x: x["input"]
            }
            | intent_prompt
            | self.low_llm
            | StrOutputParser()
        )

    # ---------------- Public Method ---------------- #
    async def process(self, user_input: str) -> dict:
        """Process user input and return bot response"""
        intent = await self.intent_chain.ainvoke({"input": user_input})

        if intent == "rag_query":
            response = await rag_chain().ainvoke({"input": user_input})
        else:
            response = await self.onboarding_chain.ainvoke({"input": user_input})
            # Extract plain text (handle dict/structured output)
        if isinstance(response, dict):
            response = (
                response.get("answer") or
                response.get("output") or
                response.get("text") or
                str(response)
            )
        else:
            response = str(response)


        # Save to shared memory
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(response)

        return {"answer": response}


# ---------------- CLI Runner ---------------- #
# async def run_cli():
#     wf = NewCustomerWorkflow()
#     print("\nRemoteLock Onboarding Assistant Ready! Type 'quit' to exit.\n")

#     while True:
#         user_input = await asyncio.to_thread(input, "You: ")
#         if user_input.strip().lower() in ["quit", "exit"]:
#             print("Bot: Goodbye!")
#             break

#         response = await wf.process(user_input)
#         print(f"Bot: {response["answer"]}")


# if __name__ == "__main__":
#     asyncio.run(run_cli())
