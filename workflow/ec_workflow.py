# src/workflow/existing_customer.py
import os
import asyncio
from src import config
from pyairtable import Table
from langchain_openai import ChatOpenAI
from workflow.rag import load_pdf_file, text_split, build_rag_pipeline
from src.memory import get_memory  # <-- import shared memory

## Rag set up
from functools import lru_cache

@lru_cache
def rag_chain():
    extracted_data = load_pdf_file("Data/")
    text_chunks = text_split(extracted_data)
    return build_rag_pipeline(text_chunks)


# ---------------- Airtable Setup ---------------- #
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")
AIRTABLE_TABLE_NAME = os.getenv("AIRTABLE_TABLE_NAME")
AIRTABLE_TOKEN = os.getenv("AIRTABLE_TOKEN")

if not all([AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME, AIRTABLE_TOKEN]):
    raise EnvironmentError("Missing Airtable environment variables.")

table = Table(AIRTABLE_TOKEN, AIRTABLE_BASE_ID, AIRTABLE_TABLE_NAME)

# ---------------- LLM & RAG Setup ---------------- #
llm_existing = ChatOpenAI(
    model=config.LOW_LLM,
    temperature=0.7,
    openai_api_key=config.OPENAI_API_KEY
)

shared_memory = get_memory()

# ---------------- Workflow Class ---------------- #
class ExistingCustomerWorkflow:
    def __init__(self):
        self.stage = "ask_email"
        self.customer_email = None
        self.open_cases = []

    async def fetch_rag_answer(self, query):
        docs = rag_chain().invoke({"input": query})
        return docs if docs else "Sorry, I could not find relevant information."

    async def process(self, user_input):
        user_input = user_input.strip().lower()
        response = ""

        if self.stage == "ask_email":
            response = "Could you please provide your email so I can check your account?"
            self.stage = "wait_email"
            return {"answer": response}

        elif self.stage == "wait_email":
            self.customer_email = user_input
            self.open_cases = table.all(formula=f"AND({{Email}}='{self.customer_email}', {{Status}}='Open')")
            if self.open_cases:
                case_list = "\n".join(f"{idx+1}. {case['fields'].get('Case Title', 'No Title')}"
                                      for idx, case in enumerate(self.open_cases))
                response = f"I found these open cases:\n{case_list}\nWould you like help with one of these cases?"
                self.stage = "help_existing_case"
            else:
                response = "No open cases found. Please choose a category:\n1. Billing\n2. Software\n3. Hardware\n4. Partner"
                self.stage = "choose_category"
            return {"answer": response}

        elif self.stage == "help_existing_case":
            if user_input in ["yes", "y", "sure", "okay"]:
                response = "Which case number would you like help with?"
                self.stage = "select_case_number"
            else:
                response = "Please choose a category instead:\n1. Billing\n2. Software\n3. Hardware\n4. Partner"
                self.stage = "choose_category"
            return {"answer": response}

        elif self.stage == "select_case_number":
            try:
                case_num = int(user_input) - 1
                chosen_case = self.open_cases[case_num]
                rag_answer = await self.fetch_rag_answer(chosen_case["fields"].get("Description", ""))
                response = f"Case Details:\n{chosen_case['fields'].get('Description', 'No Description')}\n\nSuggested solution:\n{rag_answer}"
                self.stage = "assist_case"
            except (ValueError, IndexError):
                response = "Invalid case number. Please enter a valid number."
            return {"answer": response}

        elif self.stage == "choose_category":
            categories = {"1": "Billing", "2": "Software", "3": "Hardware", "4": "Partner"}
            category_choice = categories.get(user_input)
            if category_choice:
                self.category_choice = category_choice
                response = f"Please describe your issue with {category_choice}."
                self.stage = "wait_issue_desc"
            else:
                response = "Please choose a valid option (1-4)."
            return {"answer": response}

        elif self.stage == "wait_issue_desc":
            self.issue_desc = user_input
            rag_answer = await self.fetch_rag_answer(self.issue_desc)
            response = f"Suggested information from knowledge base:\n{rag_answer}\nDo you want to create a ticket for this? (yes/no)"
            self.stage = "confirm_ticket"
            return {"answer": response}

        elif self.stage == "confirm_ticket":
            if user_input in ["yes", "y"]:
                record = table.create({
                    "Email": self.customer_email,
                    "Category": self.category_choice,
                    "Case Title": f"New {self.category_choice} Issue",
                    "Description": self.issue_desc,
                    "Status": "Open"
                })
                ticket_id = record["id"]
                response = f"Your ticket number is: TKT-{ticket_id}"
            else:
                response = "Ticket not created."
            self.stage = "done"
            return {"answer": response}

        elif self.stage == "assist_case":
            response = "Our support team will address this issue. Is there anything else I can help you with today?"
            self.stage = "ask_email"
            return {"answer": response}

        elif self.stage == "done":
            response = "Is there anything else I can help you with today?"
            self.stage = "ask_email"
            return {"answer": response}

# ---------------- CLI Runner ---------------- #
async def run_cli():
    wf = ExistingCustomerWorkflow()
    print("\nRemoteLock Existing Customer Assistant Ready! Type 'quit' to exit.\n")

    while True:
        user_input = await asyncio.to_thread(input, "You: ")
        if user_input.strip().lower() in ["quit", "exit"]:
            print("Bot: Goodbye!")
            break

        response = await wf.process(user_input)
        print(f"Bot: {response['answer']}")

if __name__ == "__main__":
    asyncio.run(run_cli())
