from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    
    information = """
    
    tell me about Elon musk and his companies"""
    
    summarization_prompt = """
    given the following {information}, summarize it in a few sentences:
    """
    
    summarization_prompt_template = PromptTemplate(
        input_variables=[information],
        template=summarization_prompt           
    )
    
    # llm = ChatOpenAI(temperature=0,model_name="gpt-5")   
    llm = ChatOllama(temperature=0, model="gemma3:270m")   

    
    chain = summarization_prompt_template | llm
    
    response = chain.invoke(input = {"information": information})
    
    # print("Hello from langchain-course!", summarization_prompt_template)
    print("Hello from langchain-course!", response)


if __name__ == "__main__":  
    main()
