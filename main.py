import cv2

cap = cv2.VideoCapture("motion-detection/runner.mp4")
ret, frame1 = cap.read() # ilk halini tut
ret, frame2 = cap.read()

while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2) # frameler arasındaki farkı bulan fonksiyon

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) # THRESH_BINARY, siyaha yakın yerleri simsiyah, beyaza yakın yerleri bembeyaz yapar.
    dilated = cv2.dilate(thresh, None, iterations= 3) # değerleri kalınlaştırmak
    
    contours,_ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1, contours, -1, (0,255,0), 3)
    
    cv2.imshow("Feed", frame1)
    
    frame1 = frame2
    ret, frame2 = cap.read()
    
    cv2.waitKey(30)
    
cap.release()
cv2.destroyAllWindows()
