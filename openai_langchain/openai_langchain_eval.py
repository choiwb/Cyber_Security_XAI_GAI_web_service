from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.embeddings_filter import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import SemanticSimilarityExampleSelector
import os
import re
import pandas as pd
import time



os.environ["OPENAI_API_KEY"] = "YOUR OPENAI API KEY !!!!!!!"

embeddings = OpenAIEmbeddings()

# chat 용 LLM
chat_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0, max_tokens=512)
# eval 용 LLM
eval_llm = ChatOpenAI(model_name='gpt-4', temperature=0, max_tokens=512)

# Evaluate Engagingness in the Dialogue Generation Task - Prompt
chat_template = """You are a cyber security analyst. about user question, answering specifically in korean.
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

eval_template = """You will be given a conversation between two cyber security analyst. You will then be given one potential
            answer for the next turn in the conversation. The answer concerns an relating cyber security fact, 
            but there are cases where it is not.
            Your task is to rate the answer on one metric.
            Please make sure you read and understand these instructions carefully. Please keep this
            conversation open while reviewing, and refer to it as needed.
            Evaluation Criteria:
            Engagingness (0-3) Is the answer as a cyber security analyst, unprofessional/WRONG/somewhat professional/very professional?
            - A score of 0 (as a cyber security analyst, unprofessional) means that the answer comparing of use the following pieces of context is generic and unprofessional (e.g., NOT related cyber security field).
            - A score of 1 (as a cyber security analyst, WRONG) means the answer comparing of use the following pieces of context is WRONG information an cyber security fact and could engage you in the conversation (e.g., WRONG number, WRONG Tactics ID, WRONG T-ID (Techniques ID)).
            - A score of 2 (as a cyber security analyst, somewhat professional) means the answer comparing of use the following pieces of context is somewhat professional and could engage you in the conversation (e.g., an opinion, thought, NOT specific number related cyber security field).
            - A score of 3 (as a cyber security analyst, very professional) means the answer comparing of use the following pieces of context is very professional or presents an cyber security fact and could engage you in the conversation ONLY specific number related cyber security field.
            Evaluation Steps:
            1. Read the conversation, the corresponding question and the answer carefully.
            2. Rate the answer on a scale of 1-3 for engagingness, according to the criteria above.
            3. Provide a brief explanation for your rating, referring to specific aspects of the answer and the conversation.
            
            context: Mitre Att&ck matrix의 Enterprise Tactics ID인 TA0001(Initial Access) 의 설명은 네트워크에 침입하기 위해 공격자가 사용하는 기술 입니다.
            question: 유튜브란 무엇이니?
            answer: 유튜브는 구글이 운영하는 동영상 공유 플랫폼입니다.
            - Engagingness: 0
            - Explanation: 검증된 문서를 기반으로 Q&A를 판단하였을 때, 사이버 보안 분야와 관련이 없는 답변입니다.
            
            context: Mitre Att&ck matrix의 Enterprise Tactics ID인 TA0006(Credential Access) 의 설명은 시스템, 도메인, 서비스 자격 증명을 접근하거나 제어하는 기술 입니다.
            question: TA0006의 영문 명을 알려줘.
            answer: TA0006의 영문 명은 initial access 입니다.
            - Engagingness: 1
            - Explanation: 검증된 문서를 기반으로 Q&A를 판단하였을 때, 사이버 보안 분야와 관련이 있는 답변이지만, 영문 명이 credential access 인데, initial access라고 답변 하였으므로 잘못된 정보 입니다.

            context: Mitre Att&ck는 공격자의 행동 기술과 기술적인 방어 기술을 분석하는 프레임 워크입니다
            question: Mitre Att&ck 에 대해서 설명해주세요.
            answer: Mitre Att&ck는 공격자의 행동 기술과 기술적인 방어 기술을 분석하는 프레임 워크입니다.
            - Engagingness: 2
            - Explanation: 검증된 문서를 기반으로 Q&A를 판단하였을 때, 사이버 보안 분야와 관련이 있는 답변이면서, 구체적인 숫자보단, 의견, 생각을 담고 있습니다.
            
            context: Mitre Att&ck matrix의 Enterprise Tactics ID는 모두 TA0043: Reconnaissance, TA0042: Resource Development, TA0001: Initial Access, TA0002: Execution, TA0003: Persistence, TA0004: Privilege Escalation, TA0005: Defense Evasion, TA0006: Credential Access, TA0007: Discovery, TA0008: Lateral Movement, TA0009: Collection, TA0011: Command and Control, TA0010: Exfiltration, TA0040: Impact 로 구성되어 있습니다.
            question: Enterprise tactics id는 모두 몇 개야?
            answer: 기업용 tactics id는 총 14개입니다.
            - Engagingness: 3
            - Explanation: 검증된 문서를 기반으로 Q&A를 판단하였을 때, 사이버 보안 분야와 관련이 있는 답변이면서, 구체적인 숫자를 포함하고 있습니다.
            
            above 3 q&a based evaluation are just examples.
            Please evaluate following Q&A comparing of use the following pieces of context.
            context: {context}
            question: {question}
            answer: {answer}
            Evaluation Form (scores ONLY)
            - Engagingness: 
            Evaluation Form (brief explanation in korean 2 sentences)
            - Explanation: """



QA_CHAIN_PROMPT_CHAT = PromptTemplate(input_variables=["context", "question"],template=chat_template)
human_message_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            template=eval_template,
            # input_variables=["question", "answer"]
            input_variables=["context", "question", "answer"]

        )
    )
eval_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])



################################################################################
# 임베딩 벡터 DB 저장 & 호출
# db_save_path = "DB SAVE PATH !!!!!!!"

new_docsearch = FAISS.load_local(os.path.join(db_save_path, 'mitre_attack_20230823_index'), embeddings)
retriever = new_docsearch.as_retriever(search_type="similarity", search_kwargs={"k":1})

# 유사도 0.7 이상만 추출
embeddings_filter = EmbeddingsFilter(embeddings = embeddings, similarity_threshold = 0.7)

# 압축 검색기 생성
compression_retriever = ContextualCompressionRetriever(base_compressor = embeddings_filter,
                                                        base_retriever = retriever)
################################################################################

# 완료된 conversation 정의 (성능 평가)
conversation = '''question: 피지컬갤러리 라는 유튜버는 누구야?
answer: 운동, 헬스를 전문적으로 알려주는 유튜버 입니다.
question: Mitre Att&ck에 대해서 설명해주세요.
answer: Mitre Att&ck는 공격자의 행동 기술과 기술적인 방어 기술을 분석하는 프레임 워크입니다. 이 프레임 워크는 공격자가 공격을 수행하는 과정을 12가지 전략으로 분류하고, 각 전략에 대한 기술적인 기법을 제공합니다. 이를 통해 조직은 자신의 보안 상태를 평가하고, 적절한 방어 전략을 수립할 수 있습니다.
question: Enterprise tactics id는 모두 몇 개야?
answer: Enterprise tactics id는 총 4개입니다.
question: 뭐 뭐 있어?
answer: Mitre Att&ck matrix의 Enterprise Tactics ID에는 다음과 같은 전략이 있습니다:
1. TA0001 (Initial Access)
2. TA0002 (Execution)
3. TA0003 (Persistence)
4. TA0004 (Privilege Escalation)
5. TA0005 (Defense Evasion)
6. TA0006 (Credential Access)
7. TA0007 (Discovery)
8. TA0008 (Lateral Movement)
9. TA0009 (Collection)
10. TA0010 (Exfiltration)
11. TA0011 (Command and Control)
12. TA0040 (Impact)
13. TA0042 (Resource Development)
14. TA0043 (Reconnaissance)
question: Command and Control의 Tactics id를 알려줘
answer: Command and Control의 Tactics ID는 TA0011입니다.
question: 위에서 내가 물어본 Tactics ID의 해당 T-ID 3개만 알려줘
answer: Tactics ID인 TA0007(Discovery)에 해당하는 T-ID는 T1016 (System Network Configuration Discovery), T1018 (Remote System Discovery), T1046 (Network Service Scanning)입니다.
question: command and control의 t-id 3개만 알려줘
answer: Command and Control의 Tactics ID인 TA0011에 해당하는 T-ID는 T1071 (Application Layer Protocol), T1071.001 (Web Protocols), T1071.002 (Standard Application Layer Protocol)입니다.
question: T1071 이건 t-id가 맞는데, t1071.001, t1071.002는 t-id의 하위 t-id 야, t-id 2개만 다른걸로 더 추천해줘
answer: T1071의 T-ID는 T1071.001 (Web Protocols)과 T1071.002 (Standard Application Layer Protocol)입니다. 다른 T-ID로는 T1071.003 (Non-Standard Port)과 T1071.004 (Standard Cryptographic Protocol)를 추천합니다.
question: 그러면 t1071에 대해서 설명해줘
answer: T1071은 Mitre Att&ck matrix에서 사용되는 Techniques ID로, 애플리케이션 레이어 프로토콜을 이용한 공격을 나타냅니다. 이 기법은 웹 프로토콜, 표준 애플리케이션 레이어 프로토콜, 비표준 포트, 표준 암호화 프로토콜 등을 통해 공격자가 목표 시스템에 대한 제어를 획득하는 방법을 의미합니다.
question: 위 t-id의 탐지 방안은?
answer: T1071 (Application Layer Protocol)의 탐지 방안은 네트워크 트래픽 분석을 통해 애플리케이션 레이어 프로토콜을 모니터링하고, 이상한 동작이나 악성 활동을 탐지하는 것입니다. T1071.001 (Web Protocols)의 탐지 방안은 웹 프로토콜을 모니터링하고, 악성 웹 트래픽이나 악성 웹사이트 접근을 탐지하는 것입니다. T1071.002 (Standard Application Layer Protocol)의 탐지 방안은 표준 애플리케이션 레이어 프로토콜을 모니터링하고, 악성 활동이나 이상한 동작을 탐지하는 것입니다.
question: 대응방안도 알려줘
answer: T1071 (Application Layer Protocol)의 대응 방안은 네트워크 트래픽 분석을 통해 애플리케이션 레이어 프로토콜을 모니터링하고, 이상한 동작이나 악성 활동을 탐지하여 차단하는 것입니다. T1071.001 (Web Protocols)의 대응 방안은 웹 프로토콜을 모니터링하고, 악성 웹 트래픽이나 악성 웹사이트 접근을 차단하는 것입니다. T1071.002 (Standard Application Layer Protocol)의 대응 방안은 표준 애플리케이션 레이어 프로토콜을 모니터링하고, 악성 활동이나 이상한 동작을 탐지하여 차단하는 것입니다.
question: garbage
answer: garbage
'''

# conversation에 \nquestion 또는 \nanswer이 아닌 \n만 있는 경우 ' ' 로 대체
conversation = re.sub(r'\n(?!(question|answer))', ' ', conversation)

eval_df = pd.DataFrame(columns = ['질문', '답변'])

'''conversation의 question: ~ answer: ~ 1 쌍 씩 추출'''
each_conversation = re.findall(r'question: (.*?)\nanswer: (.*?)\n', conversation)

formatted_conversation = []
for i in each_conversation:
    # eval_df에 i[0], i[1]을 질문, 답변 컬럼에 row 추가
    eval_df = pd.concat([eval_df, pd.DataFrame([[i[0], i[1]]], columns = ['질문', '답변'])], ignore_index=True)
    formatted_conversation.append('question: ' + i[0] + '\nanswer: ' + i[1])


conversation_history = []

def query_chain_chat(question):
    # 질문을 대화 기록에 추가
    conversation_history.append(("latest question: ", question))

    # 대화 맥락 형식화: 가장 최근의 대화만 latest question, latest answer로 나머지는 priorr question, prior answer로 표시
    if len(conversation_history) == 1:
        # print('대화 시작 !!!!!!!')
        formatted_conversation_history = f"latest question: {question}"
    else:
        formatted_conversation_history = "\n".join([f"prior answer: {text}" if sender == "latest answer: " else f"prior question: {text}" for sender, text in conversation_history])
        
        # formatted_conversation_history의 마지막 prior question은 아래 코드 에서 정의한 latest question과 동일하므로 일단 제거 필요
        lines = formatted_conversation_history.split('\n')
        if lines[-1].startswith("prior question:"):
            lines.pop()
        formatted_conversation_history = '\n'.join(lines)
        
        formatted_conversation_history += f"\nlatest question: {question}"
    # print('전체 대화 맥락 기반 질문: ', formatted_conversation_history)

    qa_chain_chat = RetrievalQA.from_chain_type(chat_llm,
                                          retriever=compression_retriever, 
                                          return_source_documents=True,
                                          chain_type_kwargs={"prompt": QA_CHAIN_PROMPT_CHAT},
                                          chain_type='stuff'
                                           )

    result = qa_chain_chat({"query": formatted_conversation_history})

    # 답변을 대화 기록에 추가 => 추 후, AIR 적용 시, DB 화 필요 함!!!!!
    conversation_history.append(("latest answer: ", result["result"]))
    
    return result['result']


def query_chain_eval(question, answer):

    context = query_chain_chat(question)
    
    qa_chain = LLMChain(llm=eval_llm, prompt=eval_prompt_template)
    result = qa_chain.run({"context": context, "question": question, "answer": answer})

    return context, result


eval_context_result = []
eval_result = []
eval_time_result = []

for i in range(eval_df.shape[0]):

    start = time.time()
    each_context, each_result = query_chain_eval(eval_df['질문'][i], eval_df['답변'][i])
    end = time.time()
    each_time = end - start
    each_time = round(each_time, 2)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print(each_context)
    eval_context_result.append(each_context)
    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print(each_result)
    eval_result.append(each_result)
    eval_time_result.append(each_time)


eval_df['질문에 대한 문서에서 유사한 context 추출'] = eval_context_result
eval_df['대화 참여도 지표 및 근거'] = eval_result
eval_df['평가 소요 시간 (초)'] = eval_time_result


eval_df.to_excel('mitre_attack_qa_eval.xlsx', index=False)

