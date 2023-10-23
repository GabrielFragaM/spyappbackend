from time import sleep
import cv2
import numpy as np
import base64
import websockets
import asyncio

largura_min = 40  # Largura mínima do retângulo
altura_min = 40   # Altura mínima do retângulo
offset = 5    # Erro permitido entre pixels
delay = 60        # FPS do vídeo
detec = []

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

subtracao = cv2.createBackgroundSubtractorKNN()

port = 5000
print("Started server on port:", port)

async def transmit(websocket, path):
    print("Client Connected!")

    # Inicialize o codec de vídeo
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    try:
        while True:
            frame_bytes = await websocket.recv()

            if frame_bytes == "END_OF_STREAM":
                break

            # Converta a lista de bytes em um objeto do tipo bytes (não uma lista de inteiros)
            frame_data = bytes(frame_bytes)

            # Use cv2.imdecode para decodificar os bytes em uma imagem
            frame1 = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)

                    
            if frame1 is not None:
                tempo = float(1 / delay)
                sleep(tempo)
                grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(grey, (3, 3), 5)
                img_sub = subtracao.apply(blur)
                dilat = cv2.dilate(img_sub, np.ones((5, 5)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
                dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
                contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                for (i, c) in enumerate(contorno):
                    (x, y, w, h) = cv2.boundingRect(c)
                    validar_contorno = (w >= largura_min) and (h >= altura_min)
                    if not validar_contorno:
                        continue

                    cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    centro = pega_centro(x, y, w, h)
                    detec.append(centro)

                cv2.imshow("Video Original", frame1)
                cv2.imshow("Detectar", dilatada)

                if cv2.waitKey(1) == 27:
                    break
    except websockets.exceptions.ConnectionClosedError:
        print("Client connection closed unexpectedly.")
    except Exception as e:
        print("An error occurred:", str(e))
    finally:
        out.release()
        cv2.destroyAllWindows()  # Certifique-se de liberar a janela OpenCV

# Altere o host para '0.0.0.0' para permitir conexões de todas as interfaces de rede
start_server = websockets.serve(transmit, host='0.0.0.0', port=port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
