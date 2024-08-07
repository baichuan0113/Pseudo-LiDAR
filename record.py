import cv2

# Open cameras
camera1 = cv2.VideoCapture(2)  # Replace 0 with the appropriate camera index for Camera 1
camera2 = cv2.VideoCapture(1)  # Replace 1 with the appropriate camera index for Camera 2

# Set video resolution (adjust as needed)
width, height = 640, 480
camera1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Define the codec and create VideoWriter objects
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output_camera1.avi', fourcc, 20.0, (width, height))
out2 = cv2.VideoWriter('output_camera2.avi', fourcc, 20.0, (width, height))

while True:
    # Capture frames from cameras
    ret1, frame1 = camera1.read()
    ret2, frame2 = camera2.read()

    if not ret1 or not ret2:
        print("Failed to capture frames")
        break

    # Display frames (optional)
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Write frames to video files
    out1.write(frame1)
    out2.write(frame2)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera1.release()
camera2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()
