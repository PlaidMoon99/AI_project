import gradio as gr
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 파일 읽기 함수
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 챗봇 응답 생성 함수
def chatbot_response(user_input, chat_history):
    # 간단한 키워드 기반 응답 예제
    if "climate" in user_input.lower() or "change" in user_input.lower():
        response = "Climate change is a critical issue affecting our planet."
    elif "history" in user_input.lower():
        response = "History is a vast subject with many interesting events to explore."
    else:
        response = "I'm here to help! Can you provide more details?"
    
    # 대화 히스토리 업데이트
    chat_history.append((user_input, response))
    return chat_history, chat_history

# 워드클라우드 생성 함수
def generate_wordcloud(chat_history):
    text = " ".join([msg[0] + " " + msg[1] for msg in chat_history])  # 입력과 응답 모두 포함
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # 워드클라우드 시각화
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig("wordcloud.png")
    return "wordcloud.png"

# 파일 내용 읽기
history_content = read_file("C:/AI_project/dataset/history.txt")
climate_change_content = read_file("C:/AI_project/dataset/Climate_Change.txt")

# Gradio 인터페이스 구성
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot with Tabs and WordCloud")

    with gr.Tab("Chatbot"):
        chat_history = gr.State([])  # 대화 히스토리 저장
        chatbot_input = gr.Textbox(label="Enter your message", placeholder="Type here...")
        chatbot_output = gr.Chatbot(label="Chat History")
        send_button = gr.Button("Send")
        
        send_button.click(chatbot_response, inputs=[chatbot_input, chat_history], outputs=[chatbot_output, chat_history])

    with gr.Tab("WordCloud"):
        generate_button = gr.Button("Generate WordCloud")
        wordcloud_output = gr.Image(label="WordCloud Visualization")
        
        generate_button.click(generate_wordcloud, inputs=[chat_history], outputs=wordcloud_output)

    with gr.Tab("File Content"):
        gr.Markdown("### History.txt")
        gr.Textbox(value=history_content, label="History Content", interactive=False, lines=10)

        gr.Markdown("### Climate_Change.txt")
        gr.Textbox(value=climate_change_content, label="Climate Change Content", interactive=False, lines=10)

# 실행
demo.launch()
