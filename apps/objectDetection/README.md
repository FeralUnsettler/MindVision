# MindVision®OD

## Código Python (por exemplo, `app.py`)

```python
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
```

---

## Documentação em Markdown (README.md)

```markdown
# Real-time Object Detection with Gemini

Esta aplicação web utiliza **Streamlit** para capturar imagens da webcam e realizar a detecção de objetos em tempo real utilizando o modelo **Gemini**. A imagem capturada é processada, enviada ao modelo que retorna caixas delimitadoras (bounding boxes) e rótulos dos objetos detectados, e, em seguida, a imagem é exibida com as anotações.

## Funcionalidades

- **Captura de Imagem:** Utiliza a webcam para capturar uma imagem via Streamlit.
- **Processamento de Imagem:** Redimensiona a imagem e a prepara para envio.
- **Integração com o Modelo Gemini:** Envia a imagem juntamente com um prompt detalhado para detectar objetos.
- **Extração e Desenho de Bounding Boxes:** Processa a resposta do modelo para extrair as caixas delimitadoras e desenhá-las na imagem.
- **Exibição dos Resultados:** Mostra a imagem anotada na interface do Streamlit.

## Pré-requisitos

- Python 3.7 ou superior
- As seguintes bibliotecas Python:
  - `opencv-python`
  - `streamlit`
  - `numpy`
  - `llama_index` (incluindo os módulos `llms.gemini` e `core.llms`)
  - Módulos padrão: `os`, `tempfile`, `re`

## Instalação

1. **Clone o repositório:**

    ```bash
    git clone https://github.com/seu_usuario/real-time-object-detection.git
    cd real-time-object-detection
    ```

2. **Crie e ative um ambiente virtual (opcional, mas recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows: venv\Scripts\activate
    ```

3. **Instale as dependências:**

    Crie um arquivo `requirements.txt` com as seguintes dependências (exemplo):

    ```txt
    opencv-python
    streamlit
    numpy
    llama_index
    ```

    Em seguida, execute:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. **Configuração da API:**

   - No arquivo `app.py`, configure a variável de ambiente `GOOGLE_API_KEY` se for necessário utilizar alguma funcionalidade que requeira a chave de API do Google.

2. **Inicie a aplicação:**

    Execute o seguinte comando para iniciar a aplicação Streamlit:

    ```bash
    streamlit run app.py
    ```

3. **Interaja com a aplicação:**

   - Abra a interface gerada pelo Streamlit no seu navegador.
   - Utilize o botão de captura para tirar uma foto com a webcam.
   - Aguarde o processamento e visualize a imagem com as detecções dos objetos.

## Estrutura do Código

- **Importações e Configurações Iniciais:**  
  Realiza as importações necessárias, configura a chave da API e inicializa o modelo Gemini.

- **Interface com Streamlit:**  
  Cria a interface de usuário com título, instruções e a captura de imagem via webcam.

- **Função `process_image`:**  
  Processa a imagem capturada:
  - Converte o buffer para um array NumPy e decodifica a imagem.
  - Redimensiona e salva temporariamente a imagem.
  - Cria e envia uma mensagem ao modelo Gemini, contendo a imagem e um prompt detalhado.
  - Extrai as caixas delimitadoras e os rótulos da resposta.
  - Desenha as caixas e os rótulos na imagem.
  
- **Processamento e Exibição:**  
  Se uma imagem é capturada, a função `process_image` é chamada e a imagem resultante é exibida na interface.

## Contribuições

Contribuições são bem-vindas! Se você deseja melhorar ou corrigir algo, sinta-se à vontade para abrir um _issue_ ou enviar um _pull request_.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
```

---

Com esses arquivos, você terá tanto o código comentado para facilitar a manutenção e entendimento quanto uma documentação completa para seu repositório no GitHub.