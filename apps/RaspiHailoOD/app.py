#!/usr/bin/env python3
import cv2
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from picamera2 import Picamera2

# ================================
# Seção: Inicialização de Variáveis e APIs
# ================================

# Configura a Google API Key (insira sua chave abaixo ou defina na variável de ambiente)
os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
if os.environ["GOOGLE_API_KEY"] == "":
    print("WARNING: GOOGLE_API_KEY não está definida. Insira sua chave de API.")

# Inicializa o modelo Gemini da Google
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ================================
# Seção: Inicialização da Câmera Oficial via Picamera2
# ================================
picam2 = Picamera2()
# Configura o preview para uma resolução de 800x600 e formato RGB
picam2.preview_configuration.main.size = (800, 600)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ================================
# Seção: Inicialização da Câmera Innomaker (Raspberry Pi GS IMX296)
# ================================
# A câmera Innomaker usa driver V4L2; normalmente está disponível em /dev/video0.
video_capture_device = 0  # Altere se o dispositivo for outro.
cap_innomaker = cv2.VideoCapture(video_capture_device)
if not cap_innomaker.isOpened():
    print("WARNING: A câmera Innomaker não pôde ser aberta via VideoCapture. Verifique a instalação do driver.")

# ================================
# Seção: Inicialização (ou Simulação) do Acelerador HAILO AI Module 8L
# ================================
# Se houver um SDK oficial do HAILO, importe e inicialize-o aqui.
# Exemplo:
# from hailo_sdk import HailoModel
# hailo_model = HailoModel("config.yaml")
#
# Para este exemplo, criamos uma função placeholder.
def hailo_process_frame(frame):
    """
    Processa o frame utilizando aceleração com o HAILO AI Module 8L.
    Esta função é um placeholder e deve ser substituída pela integração real com o SDK do HAILO.
    """
    # Exemplo de processamento: (substituir pelo processamento real)
    # processed_frame = hailo_model.infer(frame)
    # Aqui, apenas simulamos um pequeno delay para representar o tempo de inferência.
    time.sleep(0.01)
    processed_frame = frame
    return processed_frame

# ================================
# Seção: Função de Análise com o Gemini
# ================================
def analyze_image_with_gemini(image):
    """
    Converte a imagem para base64 e envia para o modelo Gemini.
    Retorna a resposta recebida.
    """
    if image is None:
        return "No image to analyze."
    
    # Converte a imagem para JPEG e para base64
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')
    
    # Cria a mensagem para o Gemini
    message = HumanMessage(
        content=[
            {"type": "text", "text": "The agent's task is to list object"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
        ]
    )
    
    # Invoca o modelo Gemini e retorna a resposta
    response = model.invoke([message])
    return response.content

# ================================
# Seção: Thread para Captura em Background (Picamera2 + Gemini)
# ================================
def background_capture_gemini():
    """
    Em intervalos regulares, captura imagens da câmera oficial (Picamera2)
    e envia para análise via Gemini.
    """
    while True:
        time.sleep(2)  # Intervalo de 2 segundos entre as capturas
        image = picam2.capture_array()
        image = cv2.flip(image, -1)  # Realiza o flip na imagem (conforme o código original)
        
        print("Enviando imagem para análise no Gemini...")
        response_content = analyze_image_with_gemini(image)
        print("Resposta do Gemini:", response_content)

# ================================
# Seção: Processamento de Vídeo em Tempo Real (Câmera Innomaker + HAILO)
# ================================
def realtime_video_processing():
    """
    Captura frames em tempo real da câmera Innomaker,
    processa cada frame utilizando aceleração HAILO e exibe o resultado.
    """
    while True:
        ret, frame = cap_innomaker.read()
        if not ret:
            print("Falha ao capturar frame da câmera Innomaker.")
            time.sleep(0.1)
            continue
        
        # Redimensiona o frame se necessário. Aqui usamos a resolução nativa 1456x1088,
        # mas ajuste conforme a necessidade de performance e exibição.
        frame = cv2.resize(frame, (1456, 1088))
        
        # Processa o frame com aceleração HAILO
        processed_frame = hailo_process_frame(frame)
        
        # Exibe o frame processado
        cv2.imshow("Feed da Câmera Innomaker (Processado HAILO)", processed_frame)
        
        # Verifica se a tecla 'q' foi pressionada para encerrar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ================================
# Seção: Função Principal
# ================================
def main():
    # Inicia a thread de background para capturas com a câmera oficial e envio ao Gemini.
    gemini_thread = threading.Thread(target=background_capture_gemini, daemon=True)
    gemini_thread.start()
    
    # Inicia o processamento de vídeo em tempo real utilizando a câmera Innomaker.
    try:
        realtime_video_processing()
    except KeyboardInterrupt:
        print("Interrompido pelo usuário. Encerrando...")
    finally:
        # Libera recursos: câmera Innomaker, janelas do OpenCV e a câmera oficial.
        if cap_innomaker.isOpened():
            cap_innomaker.release()
        cv2.destroyAllWindows()
        picam2.stop()  # Encerra a captura com a câmera oficial

if __name__ == "__main__":
    main()