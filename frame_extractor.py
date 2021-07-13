import cv2 
import argparse
import numpy as np

def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    imshape = gray_current.shape
    hsv = np.zeros((imshape[0], imshape[1], 3))

    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next.astype(np.uint8), cv2.COLOR_BGR2HSV)[:,:,1]
 
    # Flow Parameters
    # flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.1
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  

    # hue corresponds to direction
    hsv[:,:,0] = (ang * (180/ np.pi / 2)).astype(np.uint8)
    
    # value corresponds to magnitude
    mag = (mag*40).astype(np.uint8)
    mag[mag > 255] = 255
    hsv[:,:,2] = mag
    #cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to float32's
    hsv = hsv.astype(np.uint8)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    return rgb_flow

def create_frames(args) :
    cap = cv2.VideoCapture(args.video)
    i = 1 
    len_frames = 2
    
    frames = []
    while (cap.isOpened()):

        ret, frame = cap.read()
        if ret == False:
            break

        frames.append(frame)
        if len(frames) < len_frames:
            continue
        else:
            flow_frame = opticalFlowDense(frames[0], frames[-1])
            
            cv2.imwrite(f"data/{args.type}/frames/{i}.jpg", flow_frame)
            if args.view == "True":
                cv2.imshow("original (current)", frames[-1])
                cv2.imshow("original (previous)", frames[0])
                cv2.imshow("optical flow", flow_frame)
                key = cv2.waitKey(10)
            else:
                pass

            i += 1
            frames.pop(0)
            print(f"Extracting Frame {i}", end="\r")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", help="View the frames or not? True/False")
    parser.add_argument("--video", help="Video File name")
    parser.add_argument("--type", help="Path to where images will be saved")
    args = parser.parse_args()
    create_frames(args)


