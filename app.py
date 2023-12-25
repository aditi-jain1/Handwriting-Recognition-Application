import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

#Initialize our pygame
pygame.init()

WINDOWSIZEX = 640
WINDOWSIZEY = 640 
PADDING = 5
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255, 0, 9)

img_count = 1

IMAGESAVE = True
MODEL = load_model("bestmodel.h5")

LABELS = {0: "zero",
          1: "one",
          2: "two",
          3: "three",
          4: "four",
          5: "five",
          6: "six",
          7: "seven",
          8: "eight",
          9: "nine"}


screen = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

number_x = []
number_y = []

iswritting = False

predict = True

pygame.display.set_caption("Digit Board")

# Load your trained model
MODEL = load_model("bestmodel.h5")

# Load the saved image




play = True
while play:

    for event in pygame.event.get():

        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswritting:
            x, y = event.pos
            pygame.draw.circle(screen, WHITE, (x,y), 4, 0)

            number_x.append(x)
            number_y.append(y)

        if event.type == MOUSEBUTTONDOWN:
            iswritting = True
        
        if event.type == MOUSEBUTTONUP:
            iswritting = False
            number_x = sorted(number_x)
            number_y = sorted(number_y)

            rect_x1, rect_x2 = max(number_x[0]-PADDING, 0), min(WINDOWSIZEX, number_x[-1] + PADDING) 
            rect_y1, rect_y2 = max(number_y[0]-PADDING, 0), min(WINDOWSIZEY, number_y[-1] + PADDING) 

            number_x = []
            number_y = []

            img_arr = np.array(pygame.PixelArray(screen))[rect_x1:rect_x2, rect_y1:rect_y2].T.astype(np.float32)
            if IMAGESAVE:
                cv2.imwrite("image.png", img_arr)
                img_count +=1
            
            if predict:
                rect = pygame.Rect(rect_x1, rect_y1, rect_x2-rect_x1, rect_y2-rect_y1)
                sub = screen.subsurface(rect)
                pygame.image.save(sub, "screenshot.jpg")

                saved_image = cv2.imread("screenshot.jpg", cv2.IMREAD_GRAYSCALE)

                # Preprocess the image
                image = cv2.resize(saved_image, (28, 28))
                image = image.astype('float32') / 255.0  # Normalize pixel values

                # Add padding to match the training data
                image = np.pad(image, ((10, 10), (10, 10)), 'constant', constant_values=0)
                image = cv2.resize(image, (28, 28))

                # Reshape and expand dimensions for model prediction
                image_for_prediction = np.expand_dims(image, axis=0)

                # Make predictions
                prediction = MODEL.predict(image_for_prediction)
                predicted_digit = np.argmax(prediction[0])

                #Creating Visual
                pygame.draw.rect(screen, RED, rect,  2)
                font = pygame.font.Font('freesansbold.ttf', 16)
                text = font.render(str(LABELS[predicted_digit]), True, WHITE, RED)
                textRect = text.get_rect()
                textRect.center = (rect_x1, rect_y1+8)
                screen.blit(text, textRect)

                #Printing to Terminal
                print(f'Model prediction for the saved image: {predicted_digit}')
                print("image saved")
            
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    screen.fill(BLACK)
    
    pygame.display.update()