import cv2
import pytesseract
import re
import numpy as np

# Configurar o caminho para o executável do Tesseract, se necessário
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'  # Windows
# Para Linux/Mac, geralmente não é necessário, mas ajuste se precisar

def preprocess_frame(frame):
    # Converter para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro bilateral para reduzir ruído mantendo bordas
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Detectar bordas com Canny
    edged = cv2.Canny(gray, 30, 200)

    # Encontrar contornos
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        # Aproximar contorno para verificar se é um retângulo
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
        if len(approx) == 4:  # Se for um quadrilátero
            plate_contour = approx
            break

    if plate_contour is None:
        return None, gray

    # Extrair a região da placa
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate_img = gray[y:y+h, x:x+w]

    # Aplicar limiarização adaptativa para melhorar contraste
    plate_img = cv2.adaptiveThreshold(
        plate_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return plate_img, gray

def extract_plate_text(plate_img):
    # Configurar opções do Tesseract
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Extrair texto da imagem
    text = pytesseract.image_to_string(plate_img, config=custom_config)
    
    # Limpar o texto extraído
    text = text.strip().replace(" ", "").upper()
    
    # Validar formato da placa (padrão brasileiro: ABC-1234 ou ABC4D56)
    plate_pattern = r'^[A-Z]{3}-?\d{1}[A-Z0-9]{1}\d{2}$'
    if re.match(plate_pattern, text):
        # Adicionar hífen se não estiver presente
        if '-' not in text:
            text = text[:3] + '-' + text[3:]
        return text
    return None

def test_camera_indices(max_index=5):
    """Testa diferentes índices de câmera para encontrar uma webcam funcional."""
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Webcam encontrada no índice {index}")
            cap.release()
            return index
    return None

def main():
    # Testar índices de câmera
    camera_index = test_camera_indices()
    if camera_index is None:
        print("Erro: Nenhuma webcam encontrada. Verifique a conexão ou tente outro índice.")
        return

    # Iniciar captura da webcam com o índice encontrado
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print(f"Erro: Não foi possível acessar a webcam no índice {camera_index}.")
        return

    # Configurar resolução (opcional, ajuste conforme necessário)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_plate = None  # Para evitar repetição no console

    while True:
        # Capturar frame
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame. Verificando webcam...")
            break

        # Pré-processar o frame
        plate_img, gray = preprocess_frame(frame)

        # Exibir o frame original
        display_frame = frame.copy()
        
        if plate_img is not None:
            # Extrair texto da placa
            plate_text = extract_plate_text(plate_img)
            
            if plate_text:
                # Exibir o texto da placa no frame
                cv2.putText(display_frame, f"Placa: {plate_text}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Imprimir no console apenas se a placa for diferente da última detectada
                if plate_text != last_plate:
                    print(f"Placa detectada: {plate_text}")
                    last_plate = plate_text
            else:
                cv2.putText(display_frame, "Placa não reconhecida", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Exibir o frame com o resultado
        cv2.imshow('Webcam - Detecção de Placa', display_frame)

        # Sair ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()