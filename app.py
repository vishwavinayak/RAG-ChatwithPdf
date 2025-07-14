from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llm_config import get_groq_llm, get_embedding_model
from rag_pipeline import create_vector_store
from prompt import SUMMARY_PROMPT_TEMPLATE

PDF_PATH = "sample.pdf"

def main():
    """
    The main function to orchestrate the PDF summarization process.
    """
    print("--- PDF Summarization ---")

    # 1. Initialize models
    
    llm = get_groq_llm()
    embedding_model = get_embedding_model()
    
    # 2. Create the Vector Store from the PDF
    print("\nCreating vector store from the PDF...")
    vector_store = create_vector_store(PDF_PATH, embedding_model)
    
    # 3. Create a retriever to fetch relevant documents
    retriever = vector_store.as_retriever()
    
    # 4. Retrieve relevant context for a general summary query
    print("\nRetrieving relevant chunks from the PDF...")
    query = "A comprehensive summary of the entire document"
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    with open("context.txt", "w") as f:
        f.write(context)
    # return #FIXME: DEBUG
    print(f"\nRetrieved {len(relevant_docs)} relevant chunks for the context.")

    # 5. Set up the prompt and LLM chain
    prompt = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE, input_variables=["context"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # 6. Generate the summary
    print("\nGenerating summary with Groq LLM...")
    summary = llm_chain.invoke(input={"context": context})
    
    # 7. Display the result
    print("\n" + "="*50)
    print("            PDF SUMMARY")
    print("="*50 + "\n")
    print(summary['text'])
    print("\n" + "="*50)

if __name__ == "__main__":
    main()