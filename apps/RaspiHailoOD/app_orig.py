import cv2
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from picamera2 import Picamera2

# Inicializa o objeto Picamera2 para captura de imagens com a câmera oficial do Raspberry Pi.
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800,600)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# ✅ Configuração da Google API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBLUt8QsfsdgRH0sasMA52FomgThVY6Lv8"  # <== O usuário deverá inserir a chave correta.

# ✅ Inicializa o modelo Gemini da Google (modelo "gemini-1.5-flash")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Função para enviar a imagem capturada para análise no modelo Gemini.
def analyze_image_with_gemini(image):
    if image is None:
        return "No image to analyze."
    
    # Converte a imagem capturada para JPEG e, em seguida, para base64.
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')
    
    # Cria a mensagem para enviar ao Gemini.
    message = HumanMessage(
        content=[ 
            {"type": "text", "text": "The agent's task is to list object"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}} 
        ]
    )
    
    # Envia a mensagem para o Gemini e obtém a resposta.
    response = model.invoke([message])
    
    return response.content

# Função que continuamente captura imagens a cada 2 segundos e as envia para o Gemini.
def background_capture(cap):
    while True:
        time.sleep(2)  # Aguarda 2 segundos (o comentário original menciona 5s, mas o código usa 2s).
        
        im = picam2.capture_array()
        im = cv2.flip(im, -1)  # Inverte a imagem (flip em ambos os eixos)
        
        print("Sending the image for analysis...")
        response_content = analyze_image_with_gemini(im)  # Analisa a imagem com o Gemini
        print("Gemini Response: ", response_content)  # Imprime a resposta do Gemini

# Função principal para exibir continuamente o feed da webcam.
def main():
    im = picam2.capture_array()
    im = cv2.flip(im, -1)

    # Trecho comentado: verificação da abertura da câmera.
#    if not cap.isOpened():
#        print("Error: Unable to access the camera.")
#        return

    # Inicia uma thread de background para capturar e analisar imagens periodicamente.
    capture_thread = threading.Thread(target=background_capture, args=(im,))
    capture_thread.daemon = True  # Define como daemon para encerrar junto com o programa principal.
    capture_thread.start()

    # Loop principal para capturar e exibir o feed da webcam.
    while True:
        im = picam2.capture_array()
        im = cv2.flip(im, -1)
        frame = cv2.resize(im, (800,600))  # Redimensiona o frame para exibição.

        cv2.imshow("Webcam Feed", frame)

        # Sai do loop se a tecla 'q' for pressionada.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera os recursos.
    cap.release()   # **Atenção:** Aqui há um erro, pois a variável "cap" não foi definida.
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()