import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from typing import Annotated, TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from ddgs import DDGS
import textwrap

# ---------Load environment variables ---------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")

# -------------------------------
#  Define AgentState     ----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    question: str
    urls: List[str]
    raw_content: List[str]
    final_answer: Annotated[str, lambda x, y: y]

# -------------------------------
# Initialize Groq LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# ------------------------Node: Search---------------
def run_search(state: AgentState) -> AgentState:
    print("🔎 Searching for articles...")
    with DDGS() as ddgs:
        results = ddgs.text(state["question"], max_results=10)
    # Filter out social media and non-article URLs
    valid_urls = [
        r['href'] for r in results if r['href'].startswith("http")
        and not any(x in r['href'] for x in ["linkedin.com", "twitter.com", "binance.com", "facebook.com", "tiktok.com"])
    ]
    return {"urls": valid_urls[:5], "messages": [HumanMessage(content=f"Found URLs: {valid_urls[:5]}")]}

# ------------------------------- Node: Scrape and Clean -----------------------
def read_content(state: AgentState) -> AgentState:
    print(" Reading content from URLs...")
    raw_content = []
    urls_used = []

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    for url in state["urls"]:
        text = ""
        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            article_tag = soup.find("article") or soup.find("main") or soup.body
            if article_tag:
                text = article_tag.get_text(separator=" ", strip=True)
        except Exception as e:
            print(f"⚠ Failed to scrape {url}: {e}")

        if text:
            raw_content.append(text)
            urls_used.append(url)

    if not raw_content:
        print("⚠ No articles could be scraped.")
    return {"raw_content": raw_content, "urls": urls_used, "messages": [HumanMessage(content="Finished reading content.")]}

# -------------------------------
# Node:Map-Reduce Summarization
def synthesize_answer(state: AgentState) -> AgentState:
    print("🧠 Synthesizing final answer...")
    question = state["question"]
    raw_content = state["raw_content"]
    urls = state["urls"]

    if not raw_content:
        return {"final_answer": "No content could be extracted from the URLs.", "messages": [HumanMessage(content="No articles available.")]}

    # Summarize each article in chunks
    chunk_summaries = []
    for i, article in enumerate(raw_content):
        chunks = textwrap.wrap(article, 4000) 
        for j, chunk in enumerate(chunks):
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a world-class research assistant. Summarize this chunk concisely and retain the topic."),
                ("human", "Chunk {chunk_number} of Article {article_number} ({url}):\n{chunk}")
            ])
            chain = prompt | llm
            summary = chain.invoke({
                "chunk_number": j+1,
                "article_number": i+1,
                "url": urls[i],
                "chunk": chunk
            }).content
            chunk_summaries.append(f"Article {i+1} ({urls[i]}), Chunk {j+1}: {summary}")

    #  Summarize all chunk summaries into final answer
    all_summaries_text = "\n\n".join(chunk_summaries)
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world-class research assistant. Combine the following summaries into a concise answer to the user's question. Retain citations."),
        ("human", "Question: {question}\n\nSummaries:\n{summaries}")
    ])
    chain_final = final_prompt | llm
    final_answer = chain_final.invoke({"question": question, "summaries": all_summaries_text}).content

    # Save markdown report
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n**Question:** {question}\n\n")
        f.write(f"**Answer:**\n{final_answer}\n\n")
        f.write("**Sources:**\n")
        for i, url in enumerate(urls):
            f.write(f"[{i+1}] {url}\n")

    return {"final_answer": final_answer, "messages": [HumanMessage(content="Finished synthesizing. Report saved as research_report.md")]}

# -----------------Build Graph ----------------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("search", run_search)
workflow.add_node("read", read_content)
workflow.add_node("synthesize", synthesize_answer)

workflow.set_entry_point("search")
workflow.add_edge("search", "read")
workflow.add_edge("read", "synthesize")
workflow.add_edge("synthesize", END)

app = workflow.compile()


# ---------------------------------------------------------
# Run Agent
def run_agent_in_terminal(query: str):
    print("--- Starting Free Research Agent ---")
    initial_state = {"question": query, "messages": [HumanMessage(content=f"User query: {query}")]}
    for event in app.stream(initial_state):
        if "synthesize" in event and "final_answer" in event["synthesize"]:
            print("\n--- Research Complete! ---")
            print("Final Answer:")
            print(event["synthesize"]["final_answer"])
            break
        elif "messages" in event:
            last_message = event["messages"][-1]
            if last_message.content:
                print(f"Agent's thoughts: {last_message.content}")

# -------------------------------

if __name__ == "__main__":
    test_query = "What are the latest advancements in quantum computing?"
    run_agent_in_terminal(test_query)
