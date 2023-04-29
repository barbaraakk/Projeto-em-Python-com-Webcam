
import cv2
from deepface import DeepFace
import mediapipe as mp

#leitura da imagem
imagem = cv2.imread("img4.jpg")

resultado = DeepFace.analyze(imagem, actions = ["age", "emotion"])

print(resultado)

"""
webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_rosto = solucao_reconhecimento_rosto.FaceDetection()
hand = mp.solutions.hands
Hand = hand.Hands(max_num_hands=2)
desenho = mp.solutions.drawing_utils


while True:
    conseguiu_ler, frame = webcam.read()
    
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = Hand.process(imgRGB)
    handPoints = resultado.multi_hand_landmarks
    h,w,_ = frame.shape
    
    if not conseguiu_ler:
        break
    
    pontos = []
    
    if handPoints:
    
        for points in handPoints:
            #print(points)
            desenho.draw_landmarks(frame, points, hand.HAND_CONNECTIONS)
            for id,cord in enumerate(points.landmark):
                cx,cy = int(cord.x*w), int(cord.y*h)
                #cv2.putText(frame,str(id), (cx,cy+10), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)
                pontos.append([cx,cy])
                
        dedos = [8, 12, 16, 20]
        
        contador = 0
        if points:
            if pontos[4][0] > pontos[2][0]:
                contador += 1
            for x in dedos:
                if pontos[x][1] < pontos[x-2][1]:
                    contador += 1
        
        cv2.putText(frame, str(contador), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0,0), 5)
    
    cv2.waitKey(1)
    
    lista_rostos = reconhecedor_rosto.process(frame)
    
    if lista_rostos.detections:
        for rosto in lista_rostos.detections:
            desenho.draw_detection(frame, rosto)
            
    cv2.imshow("Reconhecimento", frame)
    
    
    if cv2.waitKey(5) == 27: #27 = ESC
        break
    
webcam.release()
cv2.destroyAllWindows()
"""