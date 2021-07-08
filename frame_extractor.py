import cv2 
import argparse
def create_frames(args) :
    cap = cv2.VideoCapture(args.video)
    i = 1 
    
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        
        cv2.imwrite(f"data/{args.type}/frames/{i}.jpg", frame)
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video File name")
    parser.add_argument("--type", help="Path to where images will be saved")
    args = parser.parse_args()
    create_frames(args)


