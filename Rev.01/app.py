import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import io

def find_similar_circles(image, threshold_value=1):
   # Convert the image to grayscale
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

   # Apply Gaussian blur to reduce noise
   blurred = cv2.medianBlur(gray, 13)

   _, dst = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_TOZERO)

   circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, dp=1, minDist=400,
                              param1=40, param2=10, minRadius=150, maxRadius=180)

   valid_circles = []
   if circles is not None:
       circles = np.round(circles[0, :]).astype(int)
       for (x, y, r) in circles:
           circle_mask = np.zeros_like(image)
           cv2.circle(circle_mask, (x, y), r, (255, 255, 255), -1)
           circle_roi = cv2.bitwise_and(image, circle_mask)
          
           circle_gray = cv2.cvtColor(circle_roi, cv2.COLOR_BGR2GRAY)
           _, circle_binary = cv2.threshold(circle_gray, 170, 200, cv2.THRESH_TOZERO)

           white_pixels = cv2.countNonZero(circle_binary)

           if white_pixels < 100:
               valid_circles.append((x, y, r))
              
    return np.array(valid_circles, dtype=np.uint16)

# Streamlit 페이지 설정
st.title('Image Processing App')
st.write('이 애플리케이션은 보라색 원을 찾아서 표기합니다.')

# 파일 업로드 위젯
uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True, type=['jpg', 'jpeg'])

# 이미지 처리 함수
def process_image(uploaded_file):
   # 이미지 읽기
   img = Image.open(uploaded_file)
   img = np.array(img)

   # 여기에 이미지 처리 로직을 추가하세요.
   # 예: img = find_similar_circles(img)
   img = find_similar_circles(img)

   # 처리된 이미지를 PIL 이미지로 변환
   processed_img = Image.fromarray(img)

   return processed_img

if uploaded_files:
   for uploaded_file in uploaded_files:
       # 이미지 처리
       result_img = process_image(uploaded_file)

      # 결과 보여주기
       st.image(result_img, caption='Processed Image')

      # 다운로드 링크 제공
       buffered = io.BytesIO()
       result_img.save(buffered, format="JPEG")
       st.download_button("다운로드", buffered.getvalue(), file_name="processed_image.jpg")
