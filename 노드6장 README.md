# AIFFEL Campus Online 4th Code Peer Review Templete
- 코더 : 김용석님
- 리뷰어 : 사재원


# PRT(PeerReviewTemplate)
각 항목을 스스로 확인하고 토의하여 작성한 코드에 적용합니다.
- ['X'] 1.코드가 정상적으로 동작하고 주어진 문제를 해결했나요?  
네 코드 자체는 잘 작동하였습니다만 스티커의 좌표 위치 문제가 있었습니다.  

```
#수정코드 주석부분이 원래 코드

 print (landmark[33]) # index 20
 x = landmark[33][0] # 이미지에서 코 부위의 x값 landmark[20][0]
 y = landmark[33][1] # landmark[20][1] - dlib_rect.height()//2 
 # 이미지에서 코 부위의 y값 - 얼굴 영역의 세로를 차지하는 픽셀의 수//2 → (437, 182-(186+1//2))
#landmark의 좌표가 잘못설정되어있어 스티커의 위치가 적절한 위치에 나타나지 않았습니다

refined_x = x - w // 2  #x - w // 3 
refined_y = y - h // 2 # y - h // 3
#바뀐좌표에 맞게 스티커의 중앙 부분이 x,y좌표에 맞게 스티커 크기의 절반을 뺐습니다.

img_show[refined_y:refined_y+img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,img_sticker,sticker_area).astype(np.uint8) 
# 기존 np.where(img_sticker==0,sticker_area,img_sticker)
#cv.imread 기능이 이미지를 불러올때 투명도(알파채널)가 있는 경우 투명한 부분이 255,0 둘중 하나로 입력되는데
#그 때문에 스티커 이미지가 의도와는 반대가 되는 부분이 있었습니다.

```
- [O] 2.주석을 보고 작성자의 코드가 이해되었나요?  
네 주석덕분에 전체적인 코드를 이해하는데 도움이 되었고 어떤의도에 의하여 작성한 것인지도 알 수 있어 실수인 부분들도 쉽게찾았습니다.

- [O] 3.코드가 에러를 유발할 가능성이 있나요?  
순차적으로 실행한다면 에러발생은 없었습니다.

- [O] 4.코드 작성자가 코드를 제대로 이해하고 작성했나요?  
 네 각 기능들이 어떤 역할하는지 코드리뷰를 통하여 잘 이해한 것 같습니다.
- ['X] 5.코드가 간결한가요?
함수로 묶어 여러 이미지들에 대한 출력들도 할수있으면 좋겠습니다.



# 참고 링크 및 코드 개선
```python
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
