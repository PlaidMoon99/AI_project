import gradio as gr
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tempfile
import pandas as pd
from dotenv import load_dotenv
from gtts import gTTS
from PIL import Image
import whisper
import os
import io
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


# 파일 읽기 함수
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# .env 파일에서 환경 변수 로드 (필요한 경우)
load_dotenv()

# ChatOllama 모델 초기화
model = ChatOllama(model="gemma2", temperature=0.7, verbose=False)


# 채팅 기록을 포함하여 응답을 생성하는 함수
def chat(message, history):
    # 이전 대화 기록을 ChatOllama 형식으로 변환
    chat_history = []
    for human, ai in history:
        chat_history.append(HumanMessage(content=human))
        chat_history.append(AIMessage(content=ai))

    # 현재 메시지 추가
    chat_history.append(HumanMessage(content=message))

    # 모델을 사용하여 응답 생성
    response = model.invoke(chat_history)

    return response.content

# CSV 파일 읽고 텍스트 분할하는 함수
def read_csv_and_split(file_path):
    # CSV 파일 읽기
    df = pd.read_csv(file_path, encoding='CP949')
    
    # 모든 데이터를 하나의 문자열로 합친 후, 텍스트 분할
    txt_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    txts = txt_splitter.split_text("\n".join(df.to_string(index=False, header=False)))
    
    return txts

# CSV 파일에서 질문에 대한 답을 찾는 함수
def answer_from_csv(file, question):
    # CSV 파일 읽고 텍스트 분할
    txts = read_csv_and_split(file.name)
    
    # 각 텍스트에서 질문을 찾고 답변 생성
    for txt in txts:
        if question in txt:  # 질문이 텍스트 내에 있으면
            return chat(question, [])
    return "질문에 대한 답을 찾을 수 없습니다."

# 워드클라우드 생성 함수
def generate_wordcloud(text):
    # read txt file
    with open('./dataset/history.txt', 'r', encoding='utf-8') as file:
        text = file.read()
    wordcloud = WordCloud(
        font_path='malgun',
        background_color='white',
        width=800,
        height=600,
        max_words=200,
        max_font_size=100,
        min_font_size=10,
        random_state=42
    ).generate(text)

    # 워드클라우드 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # 이미지를 메모리에서 바로 PIL 객체로 변환하여 반환
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    pil_image = Image.open(buf)
    
    return pil_image

# TTS 함수
def text_to_speech(text, lang='ko'):
    # temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        temp_filename = fp.name
        
    # TTS 변환
    tts = gTTS(text=text, lang=lang)
    tts.save(temp_filename)
    return temp_filename

def process_tts(text, lang):
    if not text:
        return None, "텍스트 입력해주세요"
    try:
        audio_file = text_to_speech(text, lang)
        return audio_file, "변환 완료! 아래에서 재생 또는 다운로드할 수 있습니다."
    except Exception as e:
        return None, f"Error!! :  {str(e)}"
    
# audio to text 함수
def transcribe_audio(audio_path): 
    # ffmpeg 경로 명시적 설정
    os.environ["PATH"] += os.pathsep + r"C:\AI_project\ffmpeg\bin"
    os.environ["FFMPEG_BINARY"] = r"C:\AI_project\ffmpeg\bin\ffmpeg.exe"
    # Loading Whisper Model
    model = whisper.load_model("base")
    # 오디오 파일 전사
    result = model.transcribe(audio_path)
    # 전사된 텍스트 반환
    return result["text"]

def process_audio(audio):
    if audio is None:
        return "Upload Audio File."
    try:
        transcribe_text = transcribe_audio(audio)
        return transcribe_text, "변환 완료!"
    except Exception as e:
        return f"Error!!: {str(e)}"
    
###########################################################################################
# 파일 내용 읽기
history_content = read_file("C:/AI_project/dataset/history.txt")
climate_change_content = read_file("C:/AI_project/dataset/Climate_Change.txt")

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot with Tabs and WordCloud")

    with gr.Tab("Chatbot"):
        chat_history = gr.State([])  # 대화 히스토리 저장
        chatbot_input = gr.Textbox(label="Enter your message", placeholder="Type here...")
        chatbot_output = gr.Textbox(label="AI Response", interactive=False)
        send_button = gr.Button("Send")

        send_button.click(chat, inputs=[chatbot_input, chat_history], outputs=[chatbot_output, chat_history])

    with gr.Tab("Examples"):
        file_input = gr.File(label="Upload CSV File with Questions")
        question_input = gr.Textbox(label="Ask a question from the file")
        response_output = gr.Textbox(label="AI Response", interactive=False)
        
           
    with gr.Tab("WordCloud"):
        # 파일 경로를 지정할 수도 있고, 직접 텍스트를 입력받을 수도 있음
        file_input = gr.File(label="Upload Text File")
        generate_button = gr.Button("Generate WordCloud")
        wordcloud_output = gr.Image(label="WordCloud Visualization")
        
        generate_button.click(generate_wordcloud, inputs=[file_input], outputs=[wordcloud_output])


    with gr.Tab("File Content"):
        gr.Markdown("### History.txt")
        gr.Textbox(value=history_content, label="History Content", interactive=False, lines=10)

        gr.Markdown("### Climate_Change.txt")
        gr.Textbox(value=climate_change_content, label="Climate Change Content", interactive=False, lines=10)

    with gr.Tab("TTS"):
        tts_input_text = gr.Textbox(lines=5, label="Input Text")
        tts_lang = gr.Dropdown(choices=['ko', 'en', 'ja', 'zh-on'], label="Select language", value='ko')
        tts_audio_output = gr.Audio(label="Generated Audio")
        tts_status = gr.Textbox(label="Status Message")
        
        tts_button = gr.Button("Generate TTS")
        tts_button.click(process_tts, inputs=[tts_input_text, tts_lang], outputs=[tts_audio_output, tts_status])
        
    with gr.Tab("Audio"):
        tts_audio_file = gr.Audio(type="filepath", label="Generated TTS Audio")  # TTS 오디오 파일 입력
        tts_text_output = gr.Textbox(lines=10, label="Transcribed Text")  # 변환된 텍스트 출력
        tts_status = gr.Textbox(label="Status Message")
        
        # 오디오 파일을 텍스트로 변환하는 버튼
        tts_button = gr.Button("Convert to Text")
        tts_button.click(process_audio, inputs=[tts_audio_file], outputs=[tts_text_output, tts_status])

# 실행
demo.launch()
