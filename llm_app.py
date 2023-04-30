import gradio as gr
import torch
from transformers import pipeline, AutoModelForCausalLM
from transformers import AutoTokenizer
import transformers

############################################
import matplotlib
matplotlib.use('agg')
############################################


MODEL = 'cerebras-gpt111m-finetune'

tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-111M")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
print(device)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    # device_map="auto",
    # load_in_8bit=True,
    # revision="8bit",
    # max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
)

pipe = pipeline(
    "text-generation",
    model=model,
    # tokenizer=MODEL,
    tokenizer=tokenizer,
    # device=2,
)

def answer(state, state_chatbot, prompt):

    ##########################################
    '''
    messages = state + [{"role": "User", "content": prompt}]

    conversation_history = "\n".join(
        # [f"### {msg['role']}:\n{msg['content']}" for msg in messages]
        [f"{msg['role']}:\n{msg['content']}" for msg in messages]

    )

    ans = pipe(
        conversation_history + "\n\nAssistant:",
        do_sample=True,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2
    )
    
    msg = ans[0]["generated_text"]
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(msg)
    '''
    ##########################################

    # prompt 에 \n, \r, \t, "가 있는 경우, ' ' 처리
    prompt = prompt.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('"', ' ')
    
    # prompt '$' 를 ' ' 로 변경
    # 주소 앞에 '$' 있을 경우 UI 표출 시 에러 발생 !!!!!
    # prompt = prompt.replace('$', ' ')

    inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"].to(model.device)
    input_ids = inputs["input_ids"]

    ##############################################
    # history 최대 512 token 까지 업데이트 코드 추가 !!!
    ##############################################
    
    generation_config = transformers.GenerationConfig(
            max_new_tokens=256,
            temperature=0.2,
            top_p=0.75,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            early_stopping=True,
            # num_beams=5,
            
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids = input_ids,
            attention_mask=torch.ones_like(input_ids),
            generation_config=generation_config
        )[0]
    

    result = tokenizer.decode(output, skip_special_tokens=True).strip()
    # print(result)
    # result에서 \n\nAssistant: 이후의 문장만 가져오기
    msg = result.split("\n\nAssistant:")[1]
    
    ##########################################

    '''
    if "\n\n" in msg:
        msg = msg.split("\n\n")[0]
    '''

    new_state = [{
        'role': 'User',
        'content': prompt
    }, {
        'role': 'Assistant',
        'content': msg
    }]
    

    state = state + new_state
    state_chatbot = state_chatbot + [(prompt, msg)]

    # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    # print(state)


    return state, state_chatbot, state_chatbot




# use via api 및 built with gradio 버튼 disable 및 제거
with gr.Blocks(css="#chatbot .overflow-y-auto{height:2000px} footer {visibility: hidden;}") as gradio_interface:

    state = gr.State([{
        'role': 'Assistant',
        'content': 'You are a Cyber Security Analyst.'
    }])
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 2000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO CHAT</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(
            container=False
        )

    txt.submit(answer, [state, state_chatbot, txt], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

gradio_interface.launch(debug=True, server_name="127.0.0.1", share=True)
