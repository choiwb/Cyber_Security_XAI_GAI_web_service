import torch
import gradio as gr
import re
import transformers
import traceback
import argparse

from queue import Queue
from threading import Thread
import gc

from peft import PeftModel, PeftConfig


############################################
import matplotlib
matplotlib.use('agg')
############################################

CUDA_AVAILABLE = torch.cuda.is_available()

device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")

# 영어 모델
# MODEL = 'cerebras-gpt111m-finetune'
# 한글 모델
# MODEL = 'koalpaca-355m-finetune'
MODEL = 'polyglot-ko-1.3b-finetune'

# tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
# tokenizer.pad_token_id = 0
# tokenizer = transformers.AutoTokenizer.from_pretrained(
#                                       "cerebras/Cerebras-GPT-111M",
#                                        max_position_embeddings = 2048,
#                                        ignore_mismatched_sizes = True)
# tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL, max_length = 2048,
#                                                       max_position_embeddings = 2048,
#                                                       ignore_mismatched_sizes = True)
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})

'''
model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, 
    max_length = 2048,
    max_position_embeddings = 2048,
    ignore_mismatched_sizes = True,
    # CUDA 존재 시만 파라미터 적용 가능 !!!!!
    load_in_8bit=True, 
    torch_dtype=torch.float16,
    device_map={'':0} if CUDA_AVAILABLE else 'auto',
)
'''

# Load peft config for pre-trained checkpoint etc.
config = PeftConfig.from_pretrained(MODEL)
print('config: ', config)
# model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  
#                            load_in_8bit=True,  torch_dtype=torch.float16, device_map={"":0})
# tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model = transformers.AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  torch_dtype = torch.float16, device_map = {'': 0}).half().to(device)
tokenizer = transformers.AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
# model = PeftModel.from_pretrained(model, MODEL, device_map={"":0})
model = PeftModel.from_pretrained(model, MODEL, torch_dtype = torch.float16, device_map = {'': 0}).half().to(device)


print(model.half())
print(model.dtype)
print(model.config)

# Streaming functionality taken from https://github.com/oobabooga/text-generation-webui/blob/master/modules/text_generation.py#L105


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """
    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc=func
        self.c_callback=callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                traceback.print_exc()
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True,None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

def clear_torch_cache():
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()

def generate_text(
    history,  
    max_new_tokens, 
    do_sample, 
    temperature, 
    top_p, 
    top_k, 
    repetition_penalty, 
    typical_p, 
    num_beams
):
    # Create a conversation context of the last 5 entries in the history
    # 학습 데이터가 적음에 따라 history가 길어질 수록 동일한 답변을 많이 유도하므로, 우선 5개 대화까지만 기억.
    inp = ''.join([
        # f"Human: {h[0]}\n\nAssistant: {'' if h[1] is None else h[1]}\n\n" for h in history[-5:]
        f"질문:{h[0]}\n답변:{'' if h[1] is None else h[1]}\n\n" for h in history[-5:]

    ]).strip()
     
    input_ids = tokenizer.encode(
        inp, 
        return_tensors='pt', 
        truncation=True, 
        add_special_tokens=False
    ).to(device) # type: ignore

    generate_params = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "typical_p": typical_p,
        "num_beams": num_beams,
        "stopping_criteria": transformers.StoppingCriteriaList(),
        "pad_token_id": tokenizer.pad_token_id,
    }

    def generate_with_callback(callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        clear_torch_cache()
        with torch.no_grad():
            model.generate(**kwargs) # type: ignore

    def generate_with_streaming(**kwargs):
        return Iteratorize(generate_with_callback, kwargs, callback=None)

    with generate_with_streaming(**generate_params) as generator:
        for output in generator:
            new_tokens = len(output) - len(input_ids[0])
            reply = tokenizer.decode(output[-new_tokens:], skip_special_tokens=True)

            # If reply contains '^Human:' or '^Assistant:' 
            # then we have reached the end of the assistant's response
            # stop_re = re.compile(r'^(Human|Assistant):', re.MULTILINE)
            stop_re = re.compile(r'^(질문|답변):', re.MULTILINE)

            if re.search(stop_re, reply):
                reply = ''.join(reply.split('\n')[:-1])
                history[-1][1] = reply.strip()
                yield history
                break

            # if reply contains 'EOS' then we have reached the end of the conversation
            if output[-1] in [tokenizer.eos_token_id]:
                yield history
                break

            history[-1][1] = reply.strip()
            yield history

with gr.Blocks(css="#chatbot .overflow-y-auto{height:2000px} footer {visibility: hidden;}") as gradio_interface:

    with gr.Row():
        gr.HTML(
        """<div style="text-align: center; max-width: 2000px; margin: 0 auto; max-height: 2000px; overflow-y: hidden;">
            <div>
                <h1>IGLOO CHAT</h1>
            </div>
        </div>"""

        )

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            # msg = gr.Textbox(value="As a Cyber Security Analyst, What is SQL Injection attack?", placeholder="Type a message...")
            msg = gr.Textbox(value="보안 전문가로서, JNDI Injection 공격에 대해서 설명해줘.", placeholder="Type a message...")

            with gr.Row():
                clear = gr.Button("Clear")

        with gr.Column():
            max_new_tokens = gr.Slider(0, 2048, 200, step=1, label="max_new_tokens")
            do_sample = gr.Checkbox(True, label="do_sample")
            with gr.Row():
                with gr.Column():
                    # 1에 가까울 수록 다양한 답변이 유도됨.
                    temperature = gr.Slider(0, 1, 0.2, step=0.01, label="temperature")
                    # 1에 가까울수록 일관된 답변이 유도됨.
                    top_p = gr.Slider(0, 1, 0.1, step=0.01, label="top_p")
                    # 답변에서 사용될 단어의 수.
                    top_k = gr.Slider(0, 100, 50, step=1, label="top_k")
                with gr.Column():
                    # 10에 가까울수록, 이미 사용된 단어는, 답변에서 사용될 단어의 등장을 억제함.
                    repetition_penalty = gr.Slider(0, 10, 1.2, step=0.01, label="repetition_penalty")
                    # 가능성 분포를 균등하게 만드는 데 사용되며, 일반적으로 1로 설정
                    typical_p = gr.Slider(0, 1, 1, step=0.01, label="typical_p")
                    # Beam Search를 수행할 때 생성할 beam의 수를 결정합니다. Beam Search는 가능한 출력의 후보군을 탐색하고, 가장 가능성이 높은 출력을 선택하는 알고리즘입니다. 
                    # 이 매개변수는 가능한 출력을 조사하는 데 필요한 시간과 메모리 양을 결정
                    num_beams = gr.Slider(0, 10, 1, step=1, label="num_beams")

    def user(user_message, history):
        # user_message 에 \n, \r, \t, "가 있는 경우, ' ' 처리
        user_message = user_message.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('"', ' ')
        
        # user_message '$' 를 ' ' 로 변경
        # 주소 앞에 '$' 있을 경우 UI 표출 시 에러 발생 !!!!!
        user_message = user_message.replace('$', ' ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
        print(user_message)

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
        generate_text, inputs=[
            chatbot,
            max_new_tokens, 
            do_sample, 
            temperature, 
            top_p, 
            top_k, 
            repetition_penalty, 
            typical_p, 
            num_beams
        ], outputs=[chatbot],
    ).then(fix_history, chatbot)

    clear.click(lambda: None, None, chatbot, queue=False)

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Chatbot Demo")
    parser.add_argument("-s", "--share", action="store_true", help="Enable sharing of the Gradio interface")
    args = parser.parse_args()

    # demo.queue().launch(share=args.share)
    demo.queue(enable_queue = True).launch(share=args.share)
'''
gradio_interface.launch(debug=True, server_name="127.0.0.1", share=True, enable_queue=True)
