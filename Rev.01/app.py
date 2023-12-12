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

               # 원을 이미지에 표시
               cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        return image, np.array(valid_circles, dtype=np.uint16)
    else:
        return image, np.array([], dtype=np.uint16)

def process_image(uploaded_file):
    img = Image.open(uploaded_file)
    img = np.array(img)

    if len(img.shape) == 3 and img.shape[2] == 3:  # 컬러 이미지인 경우
        processed_img, circles = find_similar_circles(img)
    else:
        st.error("업로드된 이미지는 컬러 이미지여야 합니다.")
        return None

    try:
        processed_img = Image.fromarray(processed_img)
        return processed_img
    except Exception as e:
        st.error(f"이미지 처리 중 오류 발생: {e}")
        return None

# Streamlit 페이지 설정
st.title('Image Processing App')
st.write('이 애플리케이션은 보라색 원을 찾아서 표기합니다.')

# 파일 업로드 위젯
uploaded_files = st.file_uploader("이미지를 업로드하세요", accept_multiple_files=True, type=['jpg', 'jpeg'])

if uploaded_files:
   for uploaded_file in uploaded_files:
       result_img = process_image(uploaded_file)
       
       if result_img:
           st.image(result_img, caption='Processed Image')
       else:
           st.write("이미지 처리에 실패했습니다.")
