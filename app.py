# Import the necessary Libraries
import os
import uuid
import json

import gradio as gr

from openai import OpenAI

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from huggingface_hub import CommitScheduler
from pathlib import Path
from dotenv import load_dotenv


# Create Client
load_dotenv()

os.environ["ANYSCALE_API_KEY"]=os.getenv("ANYSCALE_API_KEY")

client = OpenAI(
    base_url="https://api.endpoints.anyscale.com/v1",
    api_key=os.environ['ANYSCALE_API_KEY']
)

embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
# Define the embedding model and the vectorstore

collection_name = 'report-10k-2024'

vectorstore_persisted = Chroma(
    collection_name=collection_name,
    persist_directory='./report_10kdb',
    embedding_function=embedding_model
)

# Load the persisted vectorDB

retriever = vectorstore_persisted.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 5}
)


# Prepare the logging functionality

log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="RAG-investment-recommendation-log",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

# Define the Q&A system message


qna_system_message = """
You are an AI assistant to help Finsights Grey Inc., an innovative financial technology firm, develop a Retrieval-Augmented Generation (RAG) system to automate the extraction, summarization, and analysis of information from 10-K reports. Your knowledge base was last updated in August 2023.

User input will have the context required by you to answer user questions. This context will begin with the token: ###Context.
The context contains references to specific portions of a 10-K report relevant to the user query.

User questions will begin with the token: ###Question.
Your response should only be about the question asked and the context provided.
Answer only using the context provided.
Do not mention anything about the context in your final answer.
If the answer is not found in the context, it is very important for you to respond with "I don't know."
Always quote the source when you use the context. Cite the relevant source at the end of your response under the section - Source:
Do not make up sources. Use the links provided in the sources section of the context and nothing else. You are prohibited from providing other links/sources.
Here is an example of how to structure your response:

Answer:
[Answer]

Source:
[Source]
"""

# Define the user message template
qna_user_message_template = """
###Context
Here are some documents that are relevant to the question.
{context}
```
{question}
```
"""

# Define the predict function that runs when 'Submit' is clicked or when a API request is made
def predict(user_input,company):

    filter = "dataset/"+company+"-10-k-2023.pdf"
    relevant_document_chunks = vectorstore_persisted.similarity_search(user_input, k=5, filter={"source":filter})

    # Create context_for_query
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ".".join(context_list)

    # Create messages
    prompt = [
        {'role':'system', 'content': qna_system_message},
        {'role': 'user', 'content': qna_user_message_template.format(
            context=context_for_query,
            question=user_input
            )
        }
    ]

    # Get response from the LLM
    try:
        response = client.chat.completions.create(
            model='mistralai/Mixtral-8x7B-Instruct-v0.1',
            messages=prompt,
            temperature=0
        )

        prediction = response.choices[0].message.content

    except Exception as e:
        prediction = e

    # While the prediction is made, log both the inputs and outputs to a local log file
    # While writing to the log file, ensure that the commit scheduler is locked to avoid parallel
    # access

    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(json.dumps(
                {
                    'user_input': user_input,
                    'retrieved_context': context_for_query,
                    'model_response': prediction
                }
            ))
            f.write("\n")

    return prediction


examples = [
    ["What are the company's policies and frameworks regarding AI ethics, governance, and responsible AI use as detailed in their 10-K reports?", "AWS"],
    ["What are the primary business segments of the company, and how does each segment contribute to the overall revenue and profitability?", "AWS"],
    ["What are the key risk factors identified in the 10-K report that could potentially impact the company's business operations and financial performance?", "AWS"],
    ["Has the company made any significant acquisitions in the AI space, and how are these acquisitions being integrated into the company's strategy?", "Microsoft"],
    ["How much capital has been allocated towards AI research and development?","Google"],
    ["What initiatives has the company implemented to address ethical concerns surrounding AI, such as fairness, accountability, and privacy?","IBM"],
    ["How does the company plan to differentiate itself in the AI space relative to competitors?","Meta"]
]

def get_predict(question, company):
    # Implement your prediction logic here
    if company == "AWS":
        # Perform prediction for AWS
        selectedCompany = "aws"
    elif company == "IBM":
        # Perform prediction for IBM
        selectedCompany = "IBM"
    elif company == "Google":
        # Perform prediction for Google
       selectedCompany = "Google"
    elif company == "Meta":
        # Perform prediction for Meta
        selectedCompany = "meta"
    elif company == "Microsoft":
        # Perform prediction for Microsoft
        selectedCompany = "msft"
    else:
        return "Invalid company selected"

    # Implement your prediction logic here
    for example_question, example_company in examples:
        if question == example_question and selectedCompany == example_company:
            return f"This is the output for the example question: {example_question}"

    output = predict(question, selectedCompany)
    return output

# Set-up the Gradio UI
# Add text box and radio button to the interface
# The radio button is used to select the company 10k report in which the context needs to be retrieved.

# Create the interface
# For the inputs parameter of Interface provide [textbox,company]

with gr.Blocks(theme="gradio/seafoam@>=0.0.1,<0.1.0") as demo:
    with gr.Row():
        company = gr.Radio(["AWS", "IBM", "Google", "Meta", "Microsoft"], label="Select a company")

    with gr.Row():
        question = gr.Textbox(label="Enter your question")

    submit = gr.Button("Submit")
    output = gr.Textbox(label="Output")

    submit.click(
        fn=get_predict,
        inputs=[question, company],
        outputs=output
    )

    examples_component = gr.Examples(examples=examples, inputs=[question, company])

demo.queue()
demo.launch()

