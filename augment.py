import time
import cv2
import os, glob


PATH_TO_IMAGES = 'G:\\My Drive\\MLCV_nhat_Smartlab\\train_data\\train\\images'
PATH_TO_ANNATIONS = 'G:\\My Drive\\MLCV_nhat_Smartlab\\train_data\\train\\labels'

def main():
    for path_to_filename in glob.glob(os.path.join(PATH_TO_IMAGES, 'img_*')):
        img = cv2.imread(path_to_filename)
        flipped_img = cv2.flip(img, 1)
        filename = os.path.split(path_to_filename)[1]
        # cv2.imshow(filename, flip_img)
        # cv2.waitKey(0) 
        cv2.imwrite(os.path.join(PATH_TO_IMAGES, 'flipped_' + filename), flipped_img)


    for path_to_filename in glob.glob(os.path.join(PATH_TO_ANNATIONS, '*')):
        with open(path_to_filename, 'r') as f:
            lines = f.readlines()
        
        filename = os.path.split(path_to_filename)[1][:-4]
        print(filename)
        img = cv2.imread(os.path.join(PATH_TO_IMAGES, filename + '.jpg')) 
        image_width = img.shape[1]
        print(image_width)
        
        # time.sleep(10)

        flipped_lines = []
        for line in lines:
            # Parse the line to extract class index and normalized coordinates
            class_index, center_x, center_y, width, height = map(float, line.split())
            
            # Convert normalized coordinates to pixel values
            x = (center_x - width/2) * image_width
            y = center_y * image_width
            w = width * image_width
            h = height * image_width
            
            # Flip the coordinates horizontally
            x_flipped = image_width - x

            # Convert back to normalized values
            center_x_flipped = x_flipped / image_width

            # Adjust the coordinates within the image boundaries if necessary
            if center_x_flipped < 0:
                center_x_flipped = 0
            elif center_x_flipped > 1.0:
                center_x_flipped = 1.0

            # Append the flipped annotation line
            flipped_line = f"{int(class_index)} {center_x_flipped} {center_y} {width} {height}\n"
            flipped_lines.append(flipped_line)
            
        # Save the flipped annotation file
        flipped_annotation_path = os.path.join(PATH_TO_ANNATIONS, 'flipped_' + filename + '.txt')
        with open(flipped_annotation_path, 'w') as f:
            f.writelines(flipped_lines)

        print(f"Flipped annotation saved: {flipped_annotation_path}")

    print("All annotations flipped successfully!")
    

if __name__ == '__main__':
    main()