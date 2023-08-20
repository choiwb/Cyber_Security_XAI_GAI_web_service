import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re


os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY !!!!!!!"
    
doc_reader = PdfReader('SAMPLE PDF PATH !!!!!!!')


def query_chain(question):
    template = """You are a cyber security analyst. about user question, answering specifically in korean.
                Use the following pieces of context to answer the question at the end. 
                If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                Use three sentences maximum and keep the answer as concise as possible. 
                {context}
                질문: {question}
                답변: """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)

    #PDF에서 텍스트를 읽어서 raw_text변수에 저장
    raw_text = ''

    for i, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text


    #임베딩을 위해 문서를 작은 chunk로 분리해서 texts라는 변수에 나누어서 저장, chunk_overlap은 앞 chunk의 뒤에서 200까지 내용을 다시 읽어와서 저장, 즉 새 text는 800이고 200은 앞의 내용이 들어가게 됨, 숫자 바꿔서 문서에 따라 변경
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )

    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    docsearch.embedding_function
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":4})

    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7, max_tokens=512)
    
    qa_chain = RetrievalQA.from_chain_type(llm,
                                            retriever=retriever, 
                                            return_source_documents=True,
                                           chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
                                           )

    result = qa_chain({"query": question})
    return result["result"]



conversation_history = []

def chatbot_interface(user_message):
    global conversation_history
    
    # Record the user message in the conversation history
    conversation_history.append(("질문: ", user_message))

    # Implement your query_chain function or any other logic to generate the bot response
    bot_response = query_chain(user_message)

    # Record the bot response in the conversation history
    conversation_history.append(("답변: ", bot_response))

    # Construct the conversation history text
    conversation_text = "\n".join([f"{sender}: {text}" for sender, text in conversation_history])

    return conversation_text



def generate_text(history):
    # generated_history = history.copy()
    generated_history = chatbot_interface(history)

    def callback_func(reply):
        nonlocal generated_history
        
        stop_re = re.compile(r'^(질문|답변):', re.MULTILINE)
        
        if re.search(stop_re, reply):
            reply = ''.join(reply.split('\n')[:-1])
            generated_history[-1][1] = reply.strip()
            return generated_history
        
        generated_history[-1][1] = reply.strip()
        return generated_history
 
    # respomse는 최신 답변만 해당 !!!!!!!!!
    response = query_chain(generated_history[-1][0])  # Assuming the user message is the last one in history    
    
    # Call the callback function with the bot response
    generated_history = callback_func(response)

    return generated_history


            
with gr.Blocks(css="#chatbot .overflow-y-auto{height:2000px} footer {visibility: hidden;}") as gradio_interface:

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 2000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO AiR ChatBot</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(value="SQL Injection 공격에 대응하는 방법을 알려주세요.", placeholder="질문을 입력해주세요.")

            with gr.Row():
                clear = gr.Button("Clear")



    def user(user_message, history):
        # user_message 에 \n, \r, \t, "가 있는 경우, ' ' 처리
        user_message = user_message.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('"', ' ')

        return "", history + [[user_message, None]]
    
    def fix_history(history):
        update_history = False
        for i, (user, bot) in enumerate(history):
            if bot is None:
                update_history = True
                history[i][1] = "_silence_"
        if update_history:
            chatbot.update(history) 

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        # generate_text 함수의 경우, 대화의 history 를 나타냄.
        generate_text, inputs=[
            chatbot
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)


# gradio_interface.launch()
gradio_interface.launch(debug=True, server_name="127.0.0.1", share=True, enable_queue=True)
