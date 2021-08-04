from PySimpleGUI.PySimpleGUI import popup_yes_no, theme_background_color, theme_button_color, theme_element_background_color, theme_element_text_color, theme_slider_color, theme_text_color, theme_text_element_background_color
from aitextgen import aitextgen
from transformers import GPT2Tokenizer
import torch

import PySimpleGUI as sg

# yoinked from the AID2 repo, stuff to tidy up outputs
def cut_trailing_sentence(text):
    last_punc = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punc <= 0:
        last_punc = len(text) - 1

    et_token = text.find("<")
    if et_token > 0:
        last_punc = min(last_punc, et_token - 1)

    act_token = text.find(">")
    if act_token > 0:
        last_punc = min(last_punc, act_token - 1)

    text = text[:last_punc+1]

    text = cut_trailing_quotes(text)
    return text

def cut_trailing_quotes(text):
    num_quotes = text.count('"')
    if num_quotes % 2 == 0:
        return text
    else:
        final_ind = text.rfind('"')
        return text[:final_ind]


def standardize_punctuation(text):
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text

# and back to my code

ai = None
model_name = None
tokenizer = None

# default generation settings
outlen = 128
outtemp = 0.9
outreppen = 1.0
outlenpen = 1.0
outtopk = 50
outtopp = 1.0
back_memory = ''
aid_style = False

def context_window():
    print("Opening 'context' window...")
    # context window
    cont_layout = [[sg.Text('Back Memory (Inserted before output history)')],
        [sg.Multiline(default_text = back_memory, size = (120,20), key = '-MEMORY-', background_color = 'black', text_color = 'white', enable_events = True)],
                [sg.Button('Save', key = '-SAVECTX-'), sg.Text('Memory token count', key = '-CTXTOKENS-')]
    ]
    ctx_window = sg.Window("Extra stuff", cont_layout, finalize = True)
    ctx_window['-MEMORY-'].set_cursor(cursor_color = 'white')
    return ctx_window
    

def get_context_token_count(contextbox):
    print("Updating context token counts...")
    contexttokens = len(tokenizer.encode(contextbox))
    print(str(contexttokens) + " tokens in context box")
    return contexttokens

def get_token_count(inpbox, outbox):
    print("Updating token counts...")
    inptokens = len(tokenizer.encode(inpbox))
    print(str(inptokens) + " tokens in input box")
    outtokens = len(tokenizer.encode(outbox))
    print(str(outtokens) + " tokens in output box")
    return inptokens, outtokens


def generate(new_text, old_text):
    if len(back_memory)>0:
        memory = back_memory + '\n'
        memtokens = tokenizer.encode(memory)
        prompttemp = memory + old_text + " " + new_text
    else:
        prompttemp = old_text + " " + new_text
    prefixtokens = tokenizer.encode(prompttemp)
    maxlen = len(prefixtokens) + int(outlen)
    minlen = len(prefixtokens)

    prefix_token_limit = 2048 - outlen
    print("Prefix token limit: "+str(prefix_token_limit))

    print("Given prompt:\n" + prompttemp)
    if minlen > prefix_token_limit:
        print("Too many tokens! Over max limit by " + str(len(prefixtokens)-prefix_token_limit))
        leftover_tokens = prefixtokens[0:len(prefixtokens) - prefix_token_limit]
        if len(back_memory) > 0: # memory
            prefixtokens = prefixtokens[len(leftover_tokens)+len(memtokens):len(prefixtokens)]
        else: # no memory
            prefixtokens = prefixtokens[len(leftover_tokens):len(prefixtokens)]
        

    print(prefixtokens)
    prompttemp = tokenizer.decode(prefixtokens)
    print("Actual prompt being sent to the model:\n" + prompttemp)
    gen_text = ai.generate_one(prompt = prompttemp,
                                min_length = minlen,
                                max_length = maxlen,
                                temperature = outtemp,
                                repetition_penalty = outreppen,
                                length_penalty = outlenpen,
                                top_k = outtopk,
                                top_p = outtopp
                                )
    print("Generated: " + gen_text)
    torch.cuda.empty_cache()
    if len(prompttemp)>0:
        final_gen_text = gen_text[len(prompttemp)-1:]
    else:
        final_gen_text = gen_text
    final_gen_text = cut_trailing_sentence(final_gen_text)
    print("Final Generated text: " + final_gen_text)
    return final_gen_text

#instance the given model
def instance_neo(m):
    valid_choice = False
    while valid_choice == False:
        use_gpu = input("Use GPU? Y/N: ")
        if use_gpu.lower() == "y":
            use_gpu = True
            print("Using GPU. Beware of OOM.")
            valid_choice = True
        elif use_gpu.lower() == "n":
            use_gpu = False
            print("Not using GPU. Expect slow responses.")
            valid_choice = True
        continue
    valid_choice = False
    while valid_choice == False:
        use_fp16 = input("Run at half-precision? Y/N: ")
        if use_fp16.lower() == "y":
            use_fp16 = True
            print("Using FP16. Expect the occasional random token.")
            valid_choice = True
        elif use_fp16.lower() == "n":
            use_fp16 = False
            print("Not using FP16. Expect higher RAM/VRAM usage.")
            valid_choice = True
        continue
    print("Instancing model...")
    global model_name, ai, tokenizer
    if m == 1:
        print("Instancing GPT-NEO 125M!")
        model_name = 'EleutherAI/gpt-neo-125M'
    elif m == 2:
        print("Instancing GPT-NEO 1.3B!")
        model_name = 'EleutherAI/gpt-neo-1.3B'
    elif m == 3:
        print("Instancing GPT-NEO 2.7B!")
        model_name = 'EleutherAI/gpt-neo-2.7B'
    elif m == 4:
        model_name = 'TensorFlow GPT-2-124M'
        print("Instancing GPT-2 124M!")
        ai = aitextgen(tf_gpt2='124M', to_gpu=use_gpu, to_fp16=use_fp16, cache_dir='./models/gpt2-124m')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif m == 5:
        model_name = 'TensorFlow GPT-2-355M'
        print("Instancing GPT-2 355M!")
        ai = aitextgen(tf_gpt2='355M', to_gpu=use_gpu, to_fp16=use_fp16, cache_dir='./models/gpt2-355m')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif m == 6:
        model_name = 'TensorFlow GPT-2-774M'
        print("Instancing GPT-2 774M!")
        ai = aitextgen(tf_gpt2='774M', to_gpu=use_gpu, to_fp16=use_fp16, cache_dir='./models/gpt2-774m')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif m == 7:
        model_name = 'TensorFlow GPT-2-1558M'
        print("Instancing GPT-2 1558M!")
        ai = aitextgen(tf_gpt2='1558M', to_gpu=use_gpu, to_fp16=use_fp16, cache_dir='./models/gpt2-1558m')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif m == 8:
        model_name = "Custom GPT-2 model"
        print("Instancing custom model!")
        ai = aitextgen(model_folder = './models/custom', to_gpu=use_gpu, to_fp16=use_fp16, cache_dir = './models')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    elif m == 9:
        model_name = input("Enter HF Model name in format username/modelname\nExample: 'EleutherAI/gpt-neo-125M'\n> ")
        print("Instancing HF model!")
        ai = aitextgen(model = model_name, to_gpu=use_gpu, to_fp16=use_fp16, cache_dir = './models/huggingface')
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if ai == None:
        ai = aitextgen(model = model_name, to_gpu=use_gpu, to_fp16=use_fp16, cache_dir='./models/' + model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model_name

# choosing the model...
valid_choice = False
while valid_choice == False:
    m = int(input("1) GPT-NEO 125M\n2) GPT-NEO 1.3B\n3) GPT-NEO 2.7B\n4) GPT-2 124M\n5) GPT-2 355M\n6) GPT-2 774M\n7) GPT-2 1558M\n8) Custom GPT-2 Model (local, put files in the folder 'models/custom')\n9) Custom HF Model\nChoose a model...\n> "))
    if m in range(1,10):
        m = instance_neo(m)
        valid_choice = True
    else:
        print("Invalid model.")
    continue

# main window
theme_background_color('black')
theme_text_element_background_color('black')
theme_button_color('gray')
theme_text_element_background_color('black')
theme_element_text_color('white')
theme_text_color('white')
theme_slider_color('white')

# options section
opts_visible = False

options_layout = [ [sg.Text('Temperature'), sg.Slider(range = (0.01, 10.0), default_value = outtemp, resolution = 0.01, orientation = 'h', key = '-TEMP-', enable_events = True, size = (80, 16))],
                [sg.Text('Length'),sg.Slider(range = (24, 1200), default_value = outlen, resolution = 1, orientation = 'h', key = '-LENG-', enable_events = True, size = (120, 16))],
                [sg.Text('Repetition Penalty'),sg.Slider(range = (0.01, 10.0), default_value = outreppen, resolution = 0.01, orientation = 'h', key = '-REPPEN-', enable_events = True, size = (80, 16))],
                [sg.Text('Length Penalty'),sg.Slider(range = (0.01, 10.0), default_value = outlenpen, resolution = 0.01, orientation = 'h', key = '-LENPEN-', enable_events = True, size = (80, 16))],
                [sg.Text('Top K'),sg.Slider(range = (0, 100), default_value = outtopk, resolution = 1, orientation = 'h', key = '-TOPK-', enable_events = True, size = (30, 16)), sg.Text('Top P'),sg.Slider(range = (0.01, 1.0), default_value = outtopk, resolution = 0.01, orientation = 'h', key = '-TOPP-', enable_events = True, size = (30, 16))] ]

main_layout = [  [sg.Text('Yep')],
            [sg.Multiline(size = (180,24), key = '-OUTPUT-', text_color='white', background_color='black', autoscroll = True, enable_events = True)],
            [sg.Button('Clear'), sg.Button('Options'), sg.Button('Context'), sg.Text('Input box tokens: 0 / Output box tokens: 0', key = '-TOKENCOUNT-')],
            [sg.Button('Go', size = (8, 4), key = '-GO-'), sg.Frame(title = 'Options', layout = options_layout, visible = opts_visible, background_color = 'black', key = '-OPTIONS-'), sg.Multiline(size = (160, 4), key = '-INPUT-', text_color='white', background_color='black', autoscroll = True, enable_events = True)] ]

def main_loop():
    global opts_visible
    global outtemp
    global outlen
    global outreppen
    global outlenpen
    global outtopk
    global outtopp
    global back_memory
    global aid_style
    main_window, cont_win = sg.Window('Main Window', main_layout, finalize = True), None
    main_window['-INPUT-'].set_cursor(cursor_color = 'white')
    main_window['-OUTPUT-'].set_cursor(cursor_color = 'white')
    while True:
        window, event, values = sg.read_all_windows()
        # window closes
        if event == sg.WIN_CLOSED: # if user closes window
            window.close()
            print("Window closed!")
            if window == cont_win:       # if closing win 2, mark as closed
                cont_win = None
            elif window == main_window:     # if closing win 1, exit program
                break
        # generate more stuff
        elif event == '-GO-':
            print("Generate button pressed")
            print(values['-OUTPUT-'].rstrip() + " " + values['-INPUT-'].rstrip())
            print(values['-INPUT-'].rstrip())
            pref = values['-INPUT-'].rstrip()
            print(values['-OUTPUT-'].rstrip())
            if aid_style == True:
                pref = "> You " + pref
            new_gen = generate(pref, values['-OUTPUT-'].rstrip())
            if values['-OUTPUT-'].rstrip() == "":
                main_window['-OUTPUT-'].update(values['-INPUT-'].rstrip() + new_gen)
            elif values['-INPUT-'].rstrip() == "":
                main_window['-OUTPUT-'].update(values['-OUTPUT-'].rstrip() + new_gen)
            else:
                main_window['-OUTPUT-'].update(values['-OUTPUT-'].rstrip() + " " + values['-INPUT-'].rstrip() + new_gen)
            main_window['-INPUT-'].update('')
        # clear output history
        elif event == 'Clear':
            print("Clear button pressed")
            if popup_yes_no("This will clear the output history.\nAre you sure?",title="Confirm Clear", keep_on_top = True) == 'Yes':
                main_window['-INPUT-'].update('')
                main_window['-OUTPUT-'].update('')
        # toggle options visibility
        elif event == 'Options':
            print("Options button pressed")
            if opts_visible:
                opts_visible = False
                main_window['-INPUT-'].update(visible = True)
            else:
                opts_visible = True
                main_window['-INPUT-'].update(visible = False)
            main_window['-OPTIONS-'].update(visible = opts_visible)
        # open context window
        elif event == 'Context' and not cont_win:
            print("Context button pressed")
            cont_win = context_window()
        # save context
        elif event == '-SAVECTX-':
            print("Saving memory...")
            back_memory = values['-MEMORY-'].rstrip()

        # options
        elif event == '-TEMP-':
            outtemp = values['-TEMP-']
            print("Temperature updated to " + str(outtemp))
        elif event == '-LENG-':
            outlen = int(values['-LENG-'])
            print("Length updated to " + str(outlen))
        elif event == '-REPPEN-':
            outreppen = values['-REPPEN-']
            print("Rep penalty updated to " + str(outreppen))
        elif event == '-LENPEN-':
            outlenpen = values['-LENPEN-']
            print("Length penalty updated to " + str(outlenpen))
        elif event == '-TOPK-':
            outtopk = int(values['-TOPK-'])
            print("Top K updated to " + str(outtopk))
        elif event == '-TOPP-':
            outtopp = values['-TOPP-']
            print("Top P updated to " + str(outtopp))
        
        # update token counts
        if window == main_window:
            inptokens, outtokens = get_token_count(values['-INPUT-'].rstrip(), values['-OUTPUT-'].rstrip())
            new_text = 'Input box tokens: ' + str(inptokens) + ' / Output box tokens: ' + str(outtokens)
            print(new_text)
            main_window['-TOKENCOUNT-'].update(value = new_text)
        elif window == cont_win:
            ctxtokens = get_context_token_count(values['-MEMORY-'].rstrip())
            new_text_2 = 'Memory tokens: ' + str(ctxtokens)
            print(new_text_2)
            cont_win['-CTXTOKENS-'].update(value = new_text_2)
        
main_loop()