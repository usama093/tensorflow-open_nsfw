#!/usr/bin/env python
#python classify_nsfw.py -m data/open_nsfw-weights.npy test.jpg
import sys
import argparse
import tensorflow as tf
import cv2
import math

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"
flag = 0
frame_skip = 0

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
                        Only jpeg images are supported.")

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-i", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type")

    args = parser.parse_args()
    model = OpenNsfwModel()
    frameTotal=0
    frameNsfw=0
    outfilename = 'temp.avi'
    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None

        if input_type == InputType.TENSOR:
            if args.image_loader == IMAGE_LOADER_TENSORFLOW:
                fn_load_image = create_tensorflow_image_loader(tf.Session(graph=tf.Graph()))
            else:
                fn_load_image = create_yahoo_image_loader()
        elif input_type == InputType.BASE64_JPEG:
            import base64
            fn_load_image = lambda filename: np.array([base64.urlsafe_b64encode(open(filename, "rb").read())])

        sess.run(tf.global_variables_initializer())

#image = fn_load_image(args.input_file)
        videoFile = args.input_file

        cap = cv2.VideoCapture(videoFile)
        frameRate = cap.get(5) #frame rate
        ret, frame = cap.read()
        height, width, nchannels = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter( outfilename,fourcc, math.floor(frameRate), (width,height))
        global flag
        global frame_skip
        while(True):
            ret, frame = cap.read()
            if (ret != True):
                break
            else:
                if(flag):
                    frame_skip = frame_skip + 1
                    if(frame_skip == 30):
                        frame_skip = 0
                        flag = 0
                else:
                    cv2.imwrite('./images/temp.jpg', frame)
                    image = fn_load_image('./images/temp.jpg')
                    frameTotal= frameTotal+1
                    predictions = \
                        sess.run(model.predictions,
                                 feed_dict={model.input: image})
                    if(predictions[0][1]<=0.30):
                        out.write(frame)
                    else:
                        print(predictions[0][1])
                        flag= 1
                        frameNsfw= frameNsfw+1

#print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
        if(frameNsfw>0):
            print("contain sexuall content")
        else:
            print("safe")
        print((frameNsfw/frameTotal)*100)
        cap.release()
        out.release()
        
    print("Done")
if __name__ == "__main__":
    main(sys.argv)
