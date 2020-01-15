import sys
import os
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np

def detect_img(yolo, input_path, output_path='predict.txt'):
    if input_path == '':
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image, _, _, _ = yolo.detect_image(image)
                r_image.show()
    else:
        if output_path == "":
            output_path='predict.txt'

        list_image = []

        if os.path.isfile(input_path):
            with open(input_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    list_image.append(line.split()[0])
        elif os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith(".jpg"):
                    list_image.append(os.path.join("input_path", file))
        else:
            print("Input path is invalid")
            yolo.close_session()
            return

        f = open(output_path, 'w')
        for img in list_image:
            print("Process " + img)
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                line = img
                _, r_out_boxes, r_out_scores, r_out_classes = yolo.detect_image(image)

                for i in range(len(r_out_boxes)):
                    top, left, bottom, right = r_out_boxes[i]
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                    line += ' {},{},{},{},{},{},{}'.format(top, left, bottom, right,
                                                          r_out_scores[i],
                                                          r_out_classes[i],
                                                          yolo.class_names[r_out_classes[i]])
                f.write(line + '\n')

        f.close()

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors_path', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes_path', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        "--font_path", type=str,
        help='path to font, default ' + YOLO.get_defaults("font_path")
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
