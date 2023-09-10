import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI



# (회사) 유료 API 키!!!!!!!!
# 20230904_AIR	
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!!!!!!!!!!!!!!!!!!!"

data_path = 'YOUR DATA PATH !!!!!!!!!!!!!'
tactics_path = os.path.join(data_path, 'chat_gpt_context/tactics.txt')
sigmarule_yaml_sample_path = os.path.join(data_path, 'chat_gpt_context/sample_sigma_rule_yaml.txt')
snortrule_sample_path = os.path.join(data_path, 'chat_gpt_context/sample_snort_rule.txt')

def load_context(file_path):
    with open(file_path, "r") as f:
       context = f.read()

    return context



base_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            For questions, related to CVE (Common Vulnerabilities and Exposures), Please only search for CVEs released in 2015.
            For questions, related to Cyber Kill Chain Model, Please write only the names in order of all steps.
            For question, related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            payload: {payload}
            question: {question}
            answer: """
            
            
continue_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            For questions, related to CVE (Common Vulnerabilities and Exposures), Please only search for CVEs released in 2015.
            For questions, related to Cyber Kill Chain Model, Please write only the names in order of all steps.
            For question, related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            context: {context}
            payload: {payload}
            question: {question}
            answer: """
            
snort_sigma_template = """You are a cyber security analyst. about user question, answering specifically in english.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Snort Rule or Sigma Rule, Just write rule related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot, following sample Snort rule or Sigma Rule.
            Respond don't know to questions not related to cyber security.
            Keep the answer as concise as possible. 
            context: {context}
            sample snort rule or sigma rule: {sample_rule}
            payload: {payload}
            question: {question}
            answer: """
            
xai_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of XAI feature importance to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to feature importance explanation of xai, Just write based on AI detection keywords, it is associated with major attack types such as SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot following xai feature importance.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            xai feature importance: {xai_result}
            question: {question}
            answer: """


BASE_CHAIN_PROMPT = PromptTemplate(input_variables=["payload", "question"],template=base_template)
CONTINUE_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "payload", "question"],template=continue_template)
SNORT_SIGMA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "sample_rule", "payload", "question"],template=snort_sigma_template)
XAI_CHAIN_PROMPT = PromptTemplate(input_variables=["xai_result", "question"],template=xai_template)



callbacks = [StreamingStdOutCallbackHandler()]
gpt35_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.7, max_tokens=512, 
                  callbacks=callbacks, streaming=True)
gpt40_llm = ChatOpenAI(model_name='gpt-4', temperature=0.7, max_tokens=512, 
                  callbacks=callbacks, streaming=True)


base_llmchain = LLMChain(llm=gpt35_llm, prompt=BASE_CHAIN_PROMPT)
continue_llmchain = LLMChain(llm=gpt35_llm, prompt=CONTINUE_CHAIN_PROMPT)
snort_sigma_llmchain = LLMChain(llm=gpt40_llm, prompt=SNORT_SIGMA_CHAIN_PROMPT)
xai_llmchain = LLMChain(llm=gpt40_llm, prompt=XAI_CHAIN_PROMPT)



sigmarule_file = load_context(sigmarule_yaml_sample_path)
snortrule_file = load_context(snortrule_sample_path)
xai_result = '''SAMPLE XAI RESULT !!!!!!!!!!!!!!!!!!!'''


payload1 = 'SAMPLE PAYLOAD !!!!!!!!!!!!'

default_ques1 = '입력된 payload완 연관된 공격 명 1개와 판단 근거를 작성해주세요.'
default_ques2 = '피처 중요도의 AI 탐지 키워드 기반으로 설명을 작성해주세요.'
ques1 = '입력된 payload와 연관된 CVE 1개와 판단 근거를 작성해주세요.'
ques2 = '입력된 payload와 연관된 Tactics ID 1개와 판단 근거를 작성해주세요.'
ques3_0 = '입력된 payload의 Cyber Kill Chain Model을 작성해주세요.'
ques3 = '입력된 payload와 연관된 Cyber Kill Chain 대응 단계 1개 및 대응 방안을 작성해주세요.'
ques4 = '입력된 payload의 Snort Rule을 작성해주세요.'
ques5 = '입력된 payload의 Sigma Rule을 작성해주세요.'


default_ans1 = base_llmchain.run({'payload': payload1, 'question': default_ques1})
default_ans2 = xai_llmchain.run({'xai_result': xai_result, 'question': default_ques2})
ans1 = continue_llmchain.run({'context': default_ans1, 'payload': payload1, 'question': ques1})
ans2 = base_llmchain.run({'payload': payload1, 'question': ques2})
ans3_0 = base_llmchain.run({'payload': payload1, 'question': ques3_0})
ans3 = continue_llmchain.run({'context': ans3_0, 'payload': payload1, 'question': ques3})
ans4 = snort_sigma_llmchain.run({'context': default_ans1, 'sample_rule': snortrule_file, 'payload': payload1, 'question': ques4})
ans5 = snort_sigma_llmchain.run({'context': default_ans1, 'sample_rule': sigmarule_file, 'payload': payload1, 'question': ques5})

print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(default_ans1)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(default_ans2)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans1)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans2)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans3_0)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans3)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans4)
print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(ans5)