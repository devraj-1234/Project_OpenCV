import cv2
import mediapipe as mp
import time

class FaceDetector():

    def __init__(self, minDetectionCon = 0.6):                  #instancing all necessary modules to use
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw = True):                      #function to detect face and return face coordinates as a list 'bboxes'
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)

        bboxes = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img, detection)
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxes.append(id)                               #id = Face number (0 for single face and so on for multi face systems)
                bboxes.append(bbox)                             #bbox = tuple(x, y, w, h) where x = starting x coord, y = starting y coord, w = width, h = height
                                                                #So, ending x coord = x + w and ending y coord = y + h
                bboxes.append(detection.score[0])               #Face detection score - how much of a human face is detected

                if draw:                                        #To draw the bounding box around the face
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(img, f"{int(detection.score[0] * 100)}%", (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        return img, bboxes
    
    def fancyDraw(self, img, bbox, l = 30, t = 5, rt = 1):      #Thickening of corners of the bounding box (fancy stuff, nothing related to working of the program)
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        
        # Top Left  x,y
        cv2.line(img, (x, y), (x + l, y), (0, 255, 0), t)
        cv2.line(img, (x, y), (x, y + l), (0, 255, 0), t)

        # Top Right  x1,y
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)

        # Bottom Left  x,y1
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)

        # Bottom Right  x1,y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0, 255, 0), t)

        return img

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector(0.9)
    while True:
        success, img = cap.read()
        if not success:
            print("Failure")
            break
        img, bboxes = detector.findFaces(img, False)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
        cv2.imshow("Sample Test", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()