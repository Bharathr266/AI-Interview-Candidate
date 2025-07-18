from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
import os
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Get API key from env
# groq_api_key = os.getenv("GROQ_API_KEY")
groq_api_key= st.secrets["GROQ_API_KEY"]

# Load Groq LLM (Llama 3.1 model, fast + smart)
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"  # Use "llama3-70b-8192" for more powerful version if needed
)

prompt_template = ChatPromptTemplate.from_template("""
You are an AI interview assistant helping candidates prepare for interviews.

Use the provided resume context to inform your answer when relevant. If the resume contains useful information related to the question, incorporate it thoughtfully.

If the resume does not contain relevant information, you are allowed to answer using your own general knowledge — but do not make up any facts about the candidate that are not provided.

Ensure your answer is detailed, professional, and aligned with best practices for interview responses.

<context>
{context}
</context>

Question: {input}
""")


# Load the QA chain
chain = load_qa_chain(
    llm=llm,
    chain_type="stuff",
    prompt=prompt_template
)

# Final function to query Groq with full resume + notes + question
def query_groq_chain(context_text, user_question):
    docs = [Document(page_content=context_text)]

    inputs = {
        "context": context_text,
        "input": user_question
    }

    return chain.run(input_documents=docs, **inputs)
