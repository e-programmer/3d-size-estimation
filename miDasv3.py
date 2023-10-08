# Import dependencies
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

'''Select model by uncommenting model name'''
# model= 'MiDaS_small'
# model= 'DPT_Large'
model= 'DPT_Hybrid'

# Download the MiDaS
midas = torch.hub.load('intel-isl/MiDaS', model)
device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
midas.to(device)
midas.eval()

# Input transformation pipeline
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

if model =='MiDaS_small':
  transform = transforms.small_transform
else:
  transform = transforms.dpt_transform

#class to detect objects
class detectorObj():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Convert Image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create a Mask with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)


#function to estimate depth from depth map
def depth(d):
  # f= scale * depth map point + shift
  f= -0.01 * d + 1
  return f

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)

# Load Object Detector
detector = detectorObj()


#function to test code on img
#imgname should be string filename.format with valid image format eg: pod.jpeg
def testimg(imgname):
    img= cv2.imread(imgname)
    imgw, imgh, ch= img.shape
    # Transform input for midas
    dimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgbatch = transform(dimg).to('cpu')

    img= cv2.resize(img, (0,0), None, 0.25, 0.25)

    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    #Pass in custom camera calibration for accuracy
    ''' cameraMatrix: Calibration matrix from the camera calibration process.
        distCoeff: Distortion coefficients from the camera calibration process. '''
    if corners:

        # Draw polygon around the marker
        int_corners = np.int0(corners)
        cv2.polylines(img, int_corners, True, (0, 255, 0), 2)

        # Aruco Perimeter
        #first value of corners as we are only using 1 arUco marker
        aruco_perimeter = cv2.arcLength(corners[0], True)

        # Pixel to cm ratio
        pixel_cm_ratio = aruco_perimeter / 20 #The ArUco is 5x5 so it's round will be 20 (5+5+5+5)

        contours = detector.detect_objects(img)

        # Draw objects boundaries
        for cnt in contours:
            # Get rect
            rect = cv2.minAreaRect(cnt)
            (x, y), (w, h), angle = rect

            # Get Width and Height of the Objects by applying the Ratio pixel to cm
            object_width = h / pixel_cm_ratio
            object_height = w / pixel_cm_ratio

            # Display rectangle
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.polylines(img, [box], True, (255, 0, 0), 2)
            cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
            cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

            # Make a prediction
            with torch.no_grad():
                prediction = midas(imgbatch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size = img.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()

                output = prediction.cpu().numpy()
                print(output)

                #get depth map
                depth_map= output
                print(depth_map)

                #Resize depth map to target image
                depth_map= cv2.resize(depth_map, (imgw, imgh))
                depth_map= cv2.normalize(depth_map, None, 0, 1, norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_32F)

                #Determine distance to object
                depth_face= depth_map[int(x), int(y)]
                depth_face= depth(depth_face)
                print('Depth: '+ str(depth_face))

                cv2.putText(img,"Depth {} cm".format(round(102.5-(depth_face*100), 2)), (int(x - 100), int(y + 50)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

    plt.imshow(output)
    # img= cv2.resize(img, (0,0), None, 0.25, 0.25)
    cv2.imshow(img)
    plt.pause(0.00001)

        # if cv2.waitKey(10) & 0xFF == ord('q'):
            # cap.release()
            # cv2.destroyAllWindows()
    plt.show() 


'''Code for video'''

#vidname should be string either name of video, with valid video format
#Or input int 0 or 1 to use webcam
def testvideo(vidname):
    # Hook into OpenCV
    cap = cv2.VideoCapture(vidname)
    # while cap.isOpened():
    while True:
        ret, img = cap.read()

        # img= cv2.imread('frnt.jpg')
        imgw, imgh, ch= img.shape
        # Transform input for midas
        dimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgbatch = transform(dimg).to('cpu')

        img= cv2.resize(img, (0,0), None, 0.25, 0.25)

        corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
        #Pass in custom camera calibration for accuracy
        ''' cameraMatrix: Calibration matrix from the camera calibration process.
            distCoeff: Distortion coefficients from the camera calibration process. '''
        if corners:

            # Draw polygon around the marker
            int_corners = np.int0(corners)
            cv2.polylines(img, int_corners, True, (0, 255, 0), 2)

            # Aruco Perimeter
            #first value of corners as we are only using 1 arUco marker
            aruco_perimeter = cv2.arcLength(corners[0], True)

            # Pixel to cm ratio
            pixel_cm_ratio = aruco_perimeter / 20 #The ArUco is 5x5 so it's round will be 20 (5+5+5+5)

            contours = detector.detect_objects(img)

            # Draw objects boundaries
            for cnt in contours:
                # Get rect
                rect = cv2.minAreaRect(cnt)
                (x, y), (w, h), angle = rect

                # Get Width and Height of the Objects by applying the Ratio pixel to cm
                object_width = h / pixel_cm_ratio
                object_height = w / pixel_cm_ratio

                # Display rectangle
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
                cv2.polylines(img, [box], True, (255, 0, 0), 2)
                cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
                cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

                # Make a prediction
                with torch.no_grad():
                    prediction = midas(imgbatch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size = img.shape[:2],
                        mode='bicubic',
                        align_corners=False
                    ).squeeze()

                    output = prediction.cpu().numpy()
                    # print(output)

                    #get depth map
                    depth_map= output
                    # print(depth_map)

                    #Resize depth map to target image
                    depth_map= cv2.resize(depth_map, (imgw, imgh))
                    depth_map= cv2.normalize(depth_map, None, 0, 1, norm_type= cv2.NORM_MINMAX, dtype= cv2.CV_32F)

                    #Determine distance to object
                    depth_face= depth_map[int(x), int(y)]
                    depth_face= depth(depth_face)
                    print('Depth: '+ str(depth_face))

                    cv2.putText(img,"Depth {} cm".format(round(102.5-(depth_face*100), 2)), (int(x - 100), int(y + 50)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)


        # plt.imshow(output)
        # img= cv2.resize(img, (0,0), None, 0.25, 0.25)
        cv2.imshow(img)
        plt.pause(0.00001)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


# testimg('pod.jpeg')
testvideo('sample4.mp4')
# testvideo(0)