import cv2
import numpy as np
import matplotlib.pyplot as plt
model=cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def detect_img(imgpath):

    img = cv2.imread(imgpath)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face= face_cascade.detectMultiScale(img, 1.1, 4)
    x,y,w,h=face[0]
    img=img[y:y+h,x:x+w]
    img=cv2.resize(img,(224,224))
    return img

test_imgs=['test_images/test.jpg','test_images/test1.jpg','test_images/test4.jpg']

faces=[]
for imgpath in test_imgs:
    img=detect_img(imgpath)
    faces.append(img)

id=np.array([i for i in range(0,len(faces))])
model.train(faces,id)
model.save("model.yml")


histogram=model.getHistograms()
for i in range(0,len(test_imgs)):
    hist=histogram[i][0]
    axis_value=np.array([i for i in range(0,len(hist))])
    fig=plt.figure(figsize=(10,5))
    ax1=fig.add_subplot(1,2,1)
    plt.imshow(cv2.imread(test_imgs[i])[:,::,-1])
    plt.axis('off')
    ax2=fig.add_subplot(1,2,2)
    plt.bar(axis_value,hist)
    plt.show()

target='test_images/test5.jpg'
img=detect_img(target)
idx,confidence=model.predict(img)

print("found: ",test_imgs[idx])
print("confidence: ",confidence)
fig = plt.figure()

ax1 = fig.add_subplot(1, 2, 1)
# plt.imshow(img[:,:,::-1])
plt.imshow(cv2.imread(target)[:, ::-1])
plt.axis('off')

ax2 = fig.add_subplot(1, 2, 2)
# plt.imshow(faces[id], cmap='gray')
plt.imshow(cv2.imread(test_imgs[idx])[:, ::-1])
plt.axis('off')

plt.show()



