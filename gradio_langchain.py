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
    
# doc_reader = PdfReader('SAMPLE PDF PATH !!!!!!!')
with open('SAMPLE TXT PATH !!!!!!!', 'r', encoding='utf-8') as file:
    raw_text = file.read()

# raw_text = ''

# for i, page in enumerate(doc_reader.pages):
#     text = page.extract_text()
#     if text:
#         raw_text += text


text_splitter = CharacterTextSplitter(        
     # pdf 전처리가 \n\n 으로 구성됨
     separator = "\n\n",
     chunk_size = 3200,
     chunk_overlap  = 0,
     length_function = len,
)

texts = text_splitter.split_text(raw_text)


embeddings = OpenAIEmbeddings()

# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.7, max_tokens=512)
llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=512)

template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, in the case of the relationship between Tactics ID and T-ID (Techniques ID), please find T-ID (Techniques ID) based on Tactics ID.
            Tactics ID's like start 'TA' before 4 number.
            T-ID (Techniques ID) like start 'T' before 4 number.
            Tactics ID is a major category of T-ID (Techniques ID), and has an n to n relationship.
            In particular Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            {context}
            question: {question}
            answer: """


QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template)


################################################################################
# 임베딩 벡터 DB 저장 & 호출
db_save_path = "DB SAVE PATH !!!!!!!"

docsearch = FAISS.from_texts(texts, embeddings)
docsearch.embedding_function
docsearch.save_local(os.path.join(db_save_path, "mitre_attack_20230823_index"))

new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'mitre_attack_20230823_index'), embeddings)
retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":1})

# 유사도 0.7 이상만 추출
embeddings_filter = EmbeddingsFilter(embeddings = embeddings, similarity_threshold = 0.7)

# 압축 검색기 생성
compression_retriever = ContextualCompressionRetriever(base_compressor = embeddings_filter,
                                                       base_retriever = retriever)
################################################################################



conversation_history = []

def query_chain(question):
    
    # 질문을 대화 기록에 추가
    conversation_history.append(("latest question: ", question))

    # 대화 맥락 형식화: 가장 최근의 대화만 latest question, latest answer로 나머지는 priorr question, prior answer로 표시
    if len(conversation_history) == 1:
        print('대화 시작 !!!!!!!')
        formatted_conversation_history = f"latest question: {question}"
    else:
        formatted_conversation_history = "\n".join([f"prior answer: {text}" if sender == "latest answer: " else f"prior question: {text}" for sender, text in conversation_history])
        
        # formatted_conversation_history의 마지막 prior question은 아래 코드 에서 정의한 latest question과 동일하므로 일단 제거 필요
        lines = formatted_conversation_history.split('\n')
        if lines[-1].startswith("prior question:"):
            lines.pop()
        formatted_conversation_history = '\n'.join(lines)
        
        formatted_conversation_history += f"\nlatest question: {question}"
    print('전체 대화 맥락 기반 질문: ', formatted_conversation_history)

    qa_chain = RetrievalQA.from_chain_type(llm,
                                          # retriever=retriever, 
                                          retriever=compression_retriever, 

                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                                          chain_type='stuff'
                                           )

    result = qa_chain({"query": formatted_conversation_history})
    
    # 답변을 대화 기록에 추가
    conversation_history.append(("latest answer: ", result["result"]))

    return result["result"]



def generate_text(history):
    generated_history = history.copy()

    def callback_func(reply):
        nonlocal generated_history
        
        stop_re = re.compile(r'^(latest question|latest answer|prior question|prior answer):', re.MULTILINE)
        
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


            
with gr.Blocks(css="#chatbot .overflow-y-auto{height:5000px} footer {visibility: hidden;}") as gradio_interface:

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 5000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO AiR ChatBot</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            # msg = gr.Textbox(value="SQL Injection 공격에 대응하는 방법을 알려주세요.", placeholder="질문을 입력해주세요.")
            msg = gr.Textbox(value="Mitre Att&ck 에 대해서 설명해주세요.", placeholder="질문을 입력해주세요.")

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
