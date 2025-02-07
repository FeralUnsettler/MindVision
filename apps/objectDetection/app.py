import os
import cv2
import tempfile
import streamlit as st
import numpy as np
from llama_index.llms.gemini import Gemini
from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock
import re

# ----------------------------------------------------------------------
# Configuração da chave da API do Google (se necessário)
# ----------------------------------------------------------------------
# Insira sua chave de API do Google, se necessário, ou deixe em branco.
os.environ["GOOGLE_API_KEY"] = ""

# ----------------------------------------------------------------------
# Inicializa o modelo Gemini
# ----------------------------------------------------------------------
# Cria uma instância do modelo Gemini com o nome especificado.
gemini_pro = Gemini(model_name="models/gemini-1.5-flash")

# ----------------------------------------------------------------------
# Configuração da Interface com Streamlit
# ----------------------------------------------------------------------
# Define o título da aplicação e exibe uma instrução na barra lateral.
st.title("Real-time Object Detection with Gemini")
st.sidebar.write("Click 'Capture' to analyze an image.")

# ----------------------------------------------------------------------
# Captura de Imagem via Webcam
# ----------------------------------------------------------------------
# Utiliza o componente camera_input do Streamlit para capturar uma imagem.
img_file_buffer = st.camera_input("Take a picture")

# ----------------------------------------------------------------------
# Função para Processamento da Imagem
# ----------------------------------------------------------------------
def process_image(image):
    """
    Processa a imagem capturada, envia para o modelo Gemini e retorna
    a imagem anotada com caixas delimitadoras (bounding boxes) e rótulos.
    
    Parâmetros:
        image: objeto de imagem (buffer) capturado pelo Streamlit.
    
    Retorna:
        Imagem (numpy array) processada com as detecções desenhadas.
    """
    # Converte o buffer da imagem para um array NumPy
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    # Decodifica o array para obter a imagem no formato OpenCV
    img = cv2.imdecode(file_bytes, 1)

    # Redimensiona a imagem para 600x500 pixels
    img_resized = cv2.resize(img, (600, 500))
    image_height, image_width = img_resized.shape[:2]

    # Salva a imagem redimensionada em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img_resized)
        image_path = tmp.name

    # Cria a mensagem para enviar ao modelo Gemini com os blocos de texto e imagem
    msg = ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text=(
                "Return bounding boxes for Detect and return bounding boxes for all objects in the image, "
                "including people with specific attributes (e.g., person with glasses, person wearing a red shirt, "
                "person carrying a backpack, etc.), and provide details like their clothing or other features if visible. "
                "Format the output as: [ymin, xmin, ymax, xmax, object_name]. The object names should include specific descriptions "
                "(e.g., 'person with glasses', 'person in a red shirt', etc.) in the format: [ymin, xmin, ymax, xmax, object_name]. "
                "Include all objects, such as animals, vehicles, people, products and any other visible objects in the image in the format: "
                "[ymin, xmin, ymax, xmax, object_name]. Return response in text."
            )),
            ImageBlock(path=image_path, image_mimetype="image/jpeg"),
        ],
    )

    # Envia a mensagem para o modelo Gemini e obtém a resposta
    response = gemini_pro.chat(messages=[msg])

    # Utiliza uma expressão regular para extrair as caixas delimitadoras da resposta
    bounding_boxes = re.findall(r'\[(\d+,\s*\d+,\s*\d+,\s*\d+,\s*[\w\s]+)\]', response.message.content)

    list1 = []
    # Processa cada caixa encontrada na resposta
    for box in bounding_boxes:
        parts = box.split(',')
        # Converte as 4 primeiras partes para inteiros (coordenadas)
        numbers = list(map(int, parts[:-1]))
        # Obtém o rótulo do objeto
        label = parts[-1].strip()
        list1.append((numbers, label))

    # Desenha as caixas delimitadoras e os rótulos na imagem redimensionada
    for numbers, label in list1:
        ymin, xmin, ymax, xmax = numbers
        # Converte as coordenadas normalizadas (escala de 0 a 1000) para coordenadas reais da imagem
        x1 = int(xmin / 1000 * image_width)
        y1 = int(ymin / 1000 * image_height)
        x2 = int(xmax / 1000 * image_width)
        y2 = int(ymax / 1000 * image_height)

        # Desenha o retângulo (bounding box)
        cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Adiciona o texto do rótulo próximo à caixa
        cv2.putText(img_resized, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Retorna a imagem processada e anotada
    return img_resized

# ----------------------------------------------------------------------
# Processamento e Exibição da Imagem
# ----------------------------------------------------------------------
# Se uma imagem for capturada, processa e exibe a imagem anotada.
if img_file_buffer is not None:
    st.sidebar.write("Processing image... Please wait.")
    processed_img = process_image(img_file_buffer)
    
    # Converte a imagem do formato BGR (OpenCV) para RGB (para exibição correta no Streamlit)
    processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    # Exibe a imagem processada na interface
    st.image(processed_img_rgb, caption="Detected Objects", use_container_width=True)