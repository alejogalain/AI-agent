# AI-agent-using-RAG

## Main goal
The goal of this project is to build a chatbot capable of querying any PDF file, using:
- **LangChain** to extract information from the file
- **GPT-4** as the large language model (LLM)
- **Streamlit** for the user interface

## Retrieval Augmented Generation (RAG)
RAG is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge to supplement the LLM’s internal representation of information.
It has two phases: retrieval and content generation.
1. In the retrieval phase, algorithms search for and retrieve snippets of information relevant to the user’s prompt or question. This assortment of external knowledge is appended to the user’s prompt and passed to the language model. 
2. In the generative phase, the LLM draws from the augmented prompt and its internal representation of its training data to synthesize an answer tailored to the user.

#### Brief explanation of the framework
1. The user uploads a PDF file and using LangChain we extract the whole text from the file
2. As the LLM only allows a limited number of tokens to be passed as context, the text is splitted in chunks of text (smaller parts)
3. This chunks are going to be passed through an embeddings model to create a numerical representation of the text (vectorization)
4. The vectorized data is going to be stored in a vector store
5. The user's query is also going to be vectorized through the same embedding model
6. With the user's prompt the algorithm makes a semantic search of the vector store, to search for the chunks that are relevant to the query
7. The relevant chunks of text are going to be passed to our LLM as context
8. The final answer will take into account both the context provided and the user's prompt
9. In this project the final answer of the chatbot will also take into consideration the whole history of the conversation

## Installation
Clone the repository:
```
git clone [repository-link]
cd [repository-directory]
```
Install packages
```
pip install streamlit langchain langchain-openai pypdf python-dotenv chromadb
```
Use your own OPENAI_API_KEY in your .env file
```
OPENAI_API_KEY = [your-openai-api-key]
```
To run the app:
```
streamlit run main.py
```
