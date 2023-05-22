#!/usr/bin/env python
# coding: utf-8

# In[75]:


import cv2
import dlib

print(cv2.__version__)
print(dlib.__version__)


# In[76]:


# 필요한 패키지 import 하기
import os # 환경 변수나 디렉터리, 파일 등의 OS 자원을 제어할 수 있게 해주는 모듈
import cv2 # OpenCV라이브러리 → 컴퓨터 비전 관련 프로그래밍을 쉽게 할 수 있도록 도와주는 라이브러리
import matplotlib.pyplot as plt # 다양한 데이터를 많은 방법으로 도식화 할 수 있도록 하는 라이브러리
import numpy as np # 다차원 배열을 쉽게 처리하고 효율적으로 사용할 수 있도록 하는 라이브러리
import dlib # 이미지 처리 및 기계 학습, 얼굴인식 등을 할 수 있는 c++ 로 개발된 고성능의 라이브러리 
print("🌫🛸")


# In[77]:


get_ipython().system('%pwd # 경로')

my_image_path = 'images/sample_1.jpg'     # 본인 이미지가 있는 경로를 가지고 온다.
img_bgr = cv2.imread(my_image_path)    # OpenCV로 이미지를 불러옵니다
img_show = img_bgr.copy()              # 출력용 이미지를 따로 보관합니다
plt.imshow(img_bgr)                    # 이미지를 출력하기 위해 출력할 이미지를 올려준다. (실제 출력은 하지 않음)
plt.show()                             # 이미지를 출력해준다. (실제 출력)


# In[78]:


# plt.imshow 이전에 RGB 이미지로 바꾸는 것 잊지말자

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # 스퍼프에서 사람색상으로 변경
plt.imshow(img_rgb)
plt.show()


# In[79]:


# 5-3. 얼굴 검출 face detection
# dlib의 face detector는 HOG(Histogram of Oriented Gradients)와 SVM(Support Vector Machine)을 이용해 얼굴을 찾음. 
# sliding window, 큰 이미지의 작은 영역을 잘라 얼굴이 있는지 확인하고, 다시 작은 영역을 옆으로 옮겨 얼굴이 있는지 확인하는 방법


# detector 선언 

detector_hog = dlib.get_frontal_face_detector() # 기본 얼굴 감지기를 반환
print("🌫🛸")


# In[80]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# dlib은 rgb 이미지를 입력으로 받기 떄문에 cvtColor()를 이용해 opencv의 bgr이미지를 rgb로 변경

dlib_rects = detector_hog(img_rgb, 1) # 이미지 피라미드의 수, upsampling방법을 통해 크기를 키우는 이미지 피라미드

print("🌫🛸")


# In[81]:


print(dlib_rects)   

for dlib_rect in dlib_rects: # 찾은 얼굴 영역의 좌표
    l = dlib_rect.left() # 왼쪽
    t = dlib_rect.top() # 위쪽
    r = dlib_rect.right() # 오른쪽
    b = dlib_rect.bottom() # 아래쪽

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA) 
    # 시작점의 좌표와 종료점 좌표로 직각 사각형을 그림

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()

# 오 신기


# In[82]:


# 5-4. 얼굴 랜드마크 face landmark
# 이목구비의 위치를 추론하는 것을 face landmark localization 기술이라고 함.
# Object Keypoint estimation 알고리즘 

# 1. top-down : bounding box를 찾고 box내부의 keypoint를 예측
# 2. botton-up : 이미지 전체의 keypoint를 찾고 point를 이용해 군집화해서 box 생성

model_path = 'models/shape_predictor_68_face_landmarks.dat'

landmark_predictor = dlib.shape_predictor(model_path)
    # dlib 라이브러리의 shape_predictor 함수를 이용하여 모델을 불러옴
    # landmark_predictor는 RGB이미지와 dlib.rectangle를 입력 받고 dlib.full_object_detection를 반환
    # dlib.rectangle: 내부를 예측하는 박스
    # dlib.full_object_detection: 각 구성 요소의 위치와, 이미지 상의 객체의 위치를 나타냄
print("🌫🛸")


# In[83]:


list_landmarks = [] # 랜드마크의 위치를 저장할 list 생성 

# 얼굴 영역 박스마다 face landmark를 찾아내고, 좌표를 저장
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
        # 모든 landmark의 위치정보를 points 변수에 저장 
        
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
        # 각각의 landmark 위치정보를 (x, y) 형태로 변환하여 list_points 리스트로 저장
    list_landmarks.append(list_points)
        # list_landmarks에 랜드마크 리스트를 저장 
        
print(len(list_landmarks[0]))
    


# In[84]:


for landmark in list_landmarks:
    for point in landmark:
        cv2.circle(img_show, point, 2, (0, 255, 255), -1)
        
img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB) # RGB 이미지로 전환
plt.imshow(img_show_rgb)
plt.show() # 이미지 출력


# In[85]:



for dlib_rect, landmark in zip(dlib_rects, list_landmarks): 
    
    # 얼굴 영역을 저장하고 있는 값과 68개의 랜드마크를 저장하고 있는 값으로 반복문 실행
    
    print (landmark[20]) # 코의 index는 30 입니다
    x = landmark[20][0] # 이미지에서 코 부위의 x값
    y = landmark[20][1] - dlib_rect.height()//2 
    # 이미지에서 코 부위의 y값 - 얼굴 영역의 세로를 차지하는 픽셀의 수//2 → (437, 182-(186+1//2))
    
    w = h = dlib_rect.width()//2
    # 얼굴 영역의 가로를 차지하는 픽셀의 수 (531-345+1) 
    # → max(x) - min(x) +1(픽셀의 수 이기 때문에 1을 더해줌 
    # → 픽셀 수는 점 하나로도 1이 됨)
    print (f'(x,y) : ({x},{y})')
    print (f'(w,h) : ({w},{h})')


# In[86]:


sticker_path = 'images/cat-whiskers.png'
img_sticker = cv2.imread(sticker_path) 
# 스티커 이미지를 불러옵니다 // cv2.imread(이미지 경로) → image객체 행렬을 반환

img_sticker = cv2.resize(img_sticker,(w,h)) 
# 스티커 이미지 조정 → w,h는 얼굴 영역의 가로를 차지하는 픽셀의 수(187) 
# // cv2.resize(image객체 행렬, (가로 길이, 세로 길이))

print (img_sticker.shape) # 사이즈를 조정한 왕관 이미지의 차원 확인


# In[87]:


x


# In[88]:


y


# In[89]:


w


# In[90]:


h


# In[91]:


refined_x = x - w // 3 
refined_y = y - h // 3
print (f'(x,y) : ({refined_x},{refined_y})') 


# In[92]:


if refined_x < 0 :
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
    
if refined_y < 0 :
    img_sticker = img_sticker[-refined_y:, :]
    refined_y = 0
    
print (f'(x,y) : ({refined_x},{refined_y})')


# In[93]:


sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
print("슝~")


# In[94]:


plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[95]:


# 왕관 이미지
sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
# img_bgr은 7-2에서 rgb로만 적용해놓은 원본 이미지이다. 
img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) # rgb만 적용해놓은 원본 이미지에 왕관 이미지를 덮어 씌운 이미지가 나오게 된다.
plt.show()


# In[96]:


# 왕관 이미지가 이미지 밖에서 시작하지 않도록 조정이 필요함
# 좌표 순서가 y,x임에 유의한다. (y,x,rgb channel)
# 현재 상황에서는 -y 크기만큼 스티커를 crop 하고, top 의 x좌표와 y 좌표를 각각의 경우에 맞춰 원본 이미지의 경계 값으로 수정하면 아래와 같은 형식으로 나옵니다.
# 음수값 만큼 왕관 이미지(혹은 추후 적용할 스티커 이미지)를 자른다.
if refined_x < 0: 
    img_sticker = img_sticker[:, -refined_x:]
    refined_x = 0
# 왕관 이미지를 씌우기 위해 왕관 이미지가 시작할 y좌표 값 조정
if refined_y < 0:
    img_sticker = img_sticker[-refined_y:, :] # refined_y가 -98이므로, img_sticker[98: , :]가 된다. (187, 187, 3)에서 (89, 187, 3)이 됨 (187개 중에서 98개가 잘려나감)
    refined_y = 0

print (f'(x,y) : ({refined_x},{refined_y})')


# In[111]:



sticker_area = img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
print("슝~")


# In[112]:


# 왕관 이미지를 적용한 이미지를 보여준다.
# 얼굴 영역(7-3)과 랜드마크(7-4)를 미리 적용해놓은 img_show에 왕관 이미지를 덧붙인 이미지가 나오게 된다.)
plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# In[99]:



sticker_area = img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]

img_bgr[refined_y:refined_y +img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) 
plt.show()


# In[ ]:




