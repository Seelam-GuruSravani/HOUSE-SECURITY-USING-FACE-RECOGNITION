from flask import Flask, request, render_template, Response
import cv2, os
from plyer import notification
import numpy as np
from PIL import Image 

#divides the input images into small parts and plot histogram for each plot using LBP function.
recognizer = cv2.face.LBPHFaceRecognizer_create()

recognizer.read("C:/Users/laksm/Desktop/SRAVANI TOTAL/projects/ai_project/Smart-home-using-face-recognixation-main/trainer.yml")   

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
#cv2.FONT_HERSHEY_SIMPLEX is one of the font styles provided by OpenCV for adding text to images
font = cv2.FONT_HERSHEY_SIMPLEX
id = 2 
names = ['sravani','Anil K.']  
#cam = cv2.VideoCapture(0)

flag = True

app = Flask(__name__)

@app.route("/generate_frames")
def generate_frames():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) 
    cam.set(4, 480) 
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    while flag:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            #he default value of scaleFactor is 1.1, which means the image is reduced by 10% at each scale. 
            #In the code example provided, scaleFactor is set to 1.2, which means the image is reduced by 20% at each scale.
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            print(id)
            if (confidence < 60):
                #id = names[id]
                confidence = "  {0}%".format(round(100-confidence))
                print("door unlocked")
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100-confidence))
                if __name__ == "__main__":
                    notification.notify(
                        title= "Unidentified person",
                        message="Someone unidentified is at your door",
                        app_icon= "icon.ico",
                        timeout=5   )
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (2,255,0), 1)  
        cv2.imshow('camera',img) 
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")






#
@app.route("/")
def login():
    return render_template("login.html")



#post means request from web to server
@app.route("/authenticate", methods=["POST"])
def authenticate():
    name = request.form["username"]
    password = request.form["password"]
    # Authenticate the user based on their credentials
    if name == "sravani" and password == "sravani":
        return render_template("mainpage/mainpage.html")
        #return render_template("video_feed.html")
    else:
        #401= http code for unauthorised(there is mistake at browser) , 200=success 501=there is mistake at server
        return "Unauthorized", 401
    

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")







@app.route("/registerpage")
def register_page():
    return render_template("register.html")    


@app.route("/register", methods=["POST"])
def register_new_face():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480) 

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    face_id = request.form["name"]

    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    path2="C:/Users/laksm/Desktop/SRAVANI TOTAL/ai_project/Smart-home-using-face-recognixation-main/dataset"
    
    count = 0

    while(True):

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1
            name="dataset."+str(face_id) + '.' + str(count) + ".jpg"

            cv2.imwrite(os.path.join(path2, name), gray[y:y+h,x:x+w])
            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 200:
            break

    print("\n Registering completed for new face")
    #cam.release()
    return render_template("training.html")    





@app.route("/train", methods=["GET"])
def getImagesAndLabels():
    path = 'C:/Users/laksm/Desktop/SRAVANI TOTAL/ai_project/Smart-home-using-face-recognixation-main/dataset'
    

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = faceSamples, ids
    recognizer.train(faces, np.array(ids))

    recognizer.update(faces, np.array(ids))
    recognizer.write('C:/Users/laksm/Desktop/SRAVANI TOTAL/ai_project/Smart-home-using-face-recognixation-main/trainer.yml') 

    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))

    return render_template("login.html");    


#cam.release()
#cv2.destroyAllWindows()

if __name__ == "__main__":
    app.run(debug=True)
