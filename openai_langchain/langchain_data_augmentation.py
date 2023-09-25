import pandas as pd
import openai
from multiprocessing import Pool
import time
import os
from time import sleep
import traceback
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import OpenAI
from langchain.chains import LLMChain



# (회사) 유료 API 키!!!!!!!!
# 20230904_AIR
os.environ['OPENAI_API_KEY'] = "YOUR OPENAI API KEY !!!!!!!!!!!!!!!!!!!!!!!!!"


df = pd.read_excel('EDR_ChatGPT.xlsx')


system_prompt = """You are a cyber security analyst. about user question, answering specifically.
Use the following pieces of EDR log related to ransomware to answer the question at the end. 
There are 17 types of ransomware.
Shamoon ransomware type is an operational mode for encrypting data instead of overwriting it.
Maze ransomware type is a ransomware component that encrypts files with an AES key that is also RSA-1028 encrypted.
Pysa ransomware type is used RSA and AES-CBC encryption algorithm to encrypt a list of targeted file extensions.
Diavol ransomware type is encrypted files using an RSA key though the CryptEncrypt API and has appended filenames with ".lock64".
Cuba ransomware type is he ability to encrypt system data and add the ".cuba" extension to encrypted files.
REvil ransomware type is encrypt files on victim systems and demands a ransom to decrypt the files.
HELLOKITTY ransomware type is use an embedded RSA-2048 public key to encrypt victim data for ransom.
Avaddon ransomware type is encrypts the victim system using a combination of AES256 and RSA encryption schemes.
LockerGoga ransomware type is encrypted files, including core Windows OS files, using RSA-OAEP MGF1 and then demanded Bitcoin be paid for the decryption key.
DEATHRANSOM ransomware type is use public and private key pair encryption to encrypt files for ransom payment.
Conti ransomware type is use CreateIoCompletionPort(), PostQueuedCompletionStatus(), and GetQueuedCompletionPort() to rapidly encrypt files, excluding those with the extensions of .exe, .dll, and .lnk. It has used a different AES-256 encryption key per file with a bundled RAS-4096 public encryption key that is unique for each victim. Conti can use "Windows Restart Manager" to ensure files are unlocked and open for encryption.
Clop ransomware type is encrypt files using AES, RSA, and RC4 and will add the ".clop" extension to encrypted files.
Ragnar Locker ransomware type is encrypts files on the local machine and mapped drives prior to displaying a note demanding a ransom.
WastedLocker ransomware type is encrypt data and leave a ransom note.
JCry ransomware type is encrypted files and demanded Bitcoin to decrypt those files.
Bad Rabbit ransomware type is encrypted files and disks using AES-128-CBC and RSA-2048.
Seth-Locker ransomware type is encrypt files on a targeted system, appending them with the suffix .seth.

MITER ATT&CK T-IDs related to each type of ransomware are as follows.
Shamoon ransomware type's T-IDs are T1548.002, T1134.001, T1071.001, T1543.003, T1485, T1486, T1140, T1561.002, T1070.006, T1105, T1570, T1036.004, T1112, T1027, T1012, T1021.002, T1018, T1053.005, T1082, T1016, T1569.002, T1529, T1124, T1078.002.
Maze ransomware type's T-IDs are T1071.001, T1547.001, T1059.003, T1486, T1568, T1564.006, T1562.001, T1070, T1490, T1036.004, T1106, T1027, T1027.001, T1057, T1055.001, T1053.005, T1489, T1218.007, T1082, T1614.001, T1049, T1529, T1047.
Pysa ransomware type's T-IDs are T1110, T1059.001, T1059.006, T1486, T1562.001, T1070.004, T1490, T1036.005, T1112, T1046, T1003.001, T1021.001, T1489, T1016, T1569.002, T1552.001.
Diavol ransomware type's T-IDs are T1071.001, T1485, T1486, T1491.001, T1083, T1562.001, T1105, T1490, T1106, T1135, T1027, T1027.003, T1057, T1021.002, T1018, T1489, T1082, T1016, T1033.
Cuba ransomware type's T-IDs are T1134, T1059.001, T1059.003, T1543.003, T1486, T1083, T1564.003, T1070.004, T1105, T1056.001, T1036.005, T1106, T1135, T1027, T1027.002, T1057, T1620, T1489, T1082, T1614.001, T1016, T1049, T1007.
REvil ransomware type's T-IDs are T1134.001, T1134.002, T1071.001, T1059.001, T1059.003, T1059.005, T1485, T1486, T1140, T1189, T1573.002, T1041, T1083, T1562.001, T1562.009, T1070.004, T1105, T1490, T1036.005, T1112, T1106, T1027, T1027.011, T1069.002, T1566.001, T1055, T1012, T1489, T1082, T1614.001, T1007, T1204.002, T1047, T0828, T0849, T0886, T0853, T0881, T0869, T0882, T0863.
HELLOKITTY ransomware type's T-IDs are T1486, T1490, T1135, T1057, T1082, T1047.
Avaddon ransomware type's T-IDs are T1548.002, T1547.001, T1059.007, T1486, T1140, T1083, T1562.001, T1490, T1112, T1106, T1135, T1027, T1057, T1489, T1614.001, T1016, T1047.
LockerGoga ransomware type's T-IDs are T1531, T1486, T1562.001, T1070.004, T1570, T1553.002, T1529, T0827, T0828, T0829.
DEATHRANSOM ransomware type's T-IDs are T1071.001, T1486, T1083, T1105, T1490, T1135, T1082, T1614.001, T1047.
Conti ransomware type's T-IDs are T1059.003, T1486, T1140, T1083, T1490, T1106, T1135, T1027, T1057, T1055.001, T1021.002, T1018, T1489, T1016, T1049, T1080.
Clop ransomware type's T-IDs are T1059.003, T1486, T1140, T1083, T1562.001, T1490, T1112, T1106, T1135, T1027.002, T1057, T1489, T1518.001, T1553.002, T1218.007, T1614.001, T1497.003.
Ragnar Locker ransomware type's T-IDs are T1059.003, T1543.003, T1486, T1564.006, T1562.001, T1490, T1120, T1489, T1218.007, T1218.010, T1218.011, T1614, T1569.002.
WastedLocker ransomware type's T-IDs are T1548.002, T1059.003, T1543.003, T1486, T1140, T1083, T1222.001, T1564.001, T1564.004, T1574.001, T1490, T1112, T1106, T1135, T1027, T1027.001, T1120, T1012, T1569.002, T1497.001.
JCry ransomware type's T-IDs are T1547.001, T1059.001, T1059.003, T1059.005, T1486, T1490, T1204.002.
Bad Rabbit ransomware type's T-IDs are T1548.002, T1110.003, T1486, T1189, T1210, T1495, T1036.005, T1106, T1135, T1003.001, T1057, T1053.005, T1218.011, T1569.002, T1204.002, T0817, T0866, T0867, T0828, T0863.
Seth-Locker ransomware type's T-IDs are T1059.003, T1486, T1105.

<Q&A example>
<Question>
ransomware name: Clop
EDR log event detailed classification: ProcessStart
EDR log summary: explorer.exe 프로세스에 의해 3320f11728458d01eef62e10e48897ec1c2277c1fe1aa2d471a16b4dccfc1207.exe 프로세스가 시작되었습니다.
<Answer>
T-ID: T1083

Write T-ID, as concise as possible referenced above Q&A example. 
<Question>
ransomware name: {attack_type}
EDR log event detailed classification: {event_detail_classification}
EDR log summary: {event_summary}
<Answer>
T-ID: """

# 공격 유형 컬럼의 '_' 앞 만 추출
df['공격_유형_간소화'] = df['공격 유형'].str.split('_').str[0]
# print(df['공격_유형_간소화'].nunique())
# print(df['공격_유형_간소화'].value_counts())

BASE_CHAIN_PROMPT = PromptTemplate(input_variables=["attack_type", "event_detail_classification", "event_summary"],template=system_prompt)

callbacks = [StreamingStdOutCallbackHandler()]

# LLM
llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0, max_tokens=1024, 
                  callbacks=callbacks, streaming=True)

qa_chain = LLMChain(llm=llm, prompt=BASE_CHAIN_PROMPT)

df['GPT_T-ID'] = ''

aug_df = pd.DataFrame()

# Calculate the number of DataFrames to create
each_sampling_df_len = 100
num_dataframes = (len(df) // each_sampling_df_len) + 1 if len(df) % each_sampling_df_len != 0 else len(df) // each_sampling_df_len

# Split the DataFrame into smaller DataFrames and create individual variables
for i in range(num_dataframes):
    start_idx = i * each_sampling_df_len
    end_idx = min((i + 1) * each_sampling_df_len, len(df))
    globals()[f'df{i+1}'] = df[start_idx:end_idx].copy()
    print(globals()[f'df{i+1}'].shape)


def chatgpt_orca_answer(row):
    index, data = row
    attack_type = data['공격_유형_간소화']
    event_detail_classification = data['이벤트 상세 분류']
    event_summary = data['이벤트 요약']
    result = qa_chain.run({"attack_type": attack_type, "event_detail_classification": event_detail_classification, "event_summary": event_summary})
    
    return result



def process_row(row):
    index, data = row

    try:
        completion = chatgpt_orca_answer(row)

        answer = completion.lower().replace('\n', ' ')
        # print(index)
        # print('-------answer-------')
        # print(answer)
        return answer
    except openai.error.RateLimitError as e:
        # Handle rate limit error here (e.g., wait for 1 minute and retry)
        print("Rate limit error: ", e)
        print("Waiting for 1 minute before retrying...")
        # time.sleep(60)  # Wait for 1 minute (60 seconds)
        # print("Retrying...")
        # return process_row(row)  # Retry the API call recursively
        pass


def process_row_with_retry(row, max_retries=3, sleep_interval=60):
    for i in range(max_retries):
        try:
            return process_row(row)
        except openai.error.APIError as e:
            print(f"Error occurred: {e}. Retrying ({i+1}/{max_retries}) after {sleep_interval} seconds...")
            sleep(sleep_interval)  # prevent too many requests in a short time
        except Exception as e:  # catch other exceptions
            print(f"Unexpected error occurred: {traceback.format_exc()}")
    return None  # or you may want to return a special value indicating error

start = time.time()

for i in range(num_dataframes):

    with Pool(int(os.cpu_count() / 4)) as pool:
        answers = pool.map(process_row_with_retry, globals()[f'df{i+1}'].iterrows())

    globals()[f'df{i+1}']['GPT_T-ID'] = answers    
    print('%d번 째 100개 데이터셋 답변 완료' %(i+1))
    # aug_df 에 globals()[f'df{i+1}'] 를 concat
    aug_df = pd.concat([aug_df, globals()[f'df{i+1}']], axis=0)
    print(aug_df.shape)

end = time.time()
print('답변 생성 소요 시간: %.2f (초)' %(end - start))



aug_df.to_excel('EDR_ChatGPT_label.xlsx', index=False)
