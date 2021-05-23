import pygame
import cv2
import numpy as np
from model import model
import tensorflow as tf

pygame.init()

# Creating a window
win = pygame.display.set_mode((448, 478))
pygame.display.set_caption("Number Guesser")

win.fill((255, 255, 255))

# Getting sizes of the window
width = win.get_width()
height = win.get_height()

# Setting up fonts
font = pygame.font.SysFont("Helvetica", 15)
result_font = pygame.font.SysFont("Helvetica", 30)
ok_text = font.render("OK", True, (0, 0, 0))
erase_text = font.render("Erase", True, (0, 0, 0))

# Part of the screen to be saved
rect = pygame.Rect(0, 30, width, height - 30)


def ok_pressed():
    # Saving a part of the screen with a number
    sub = win.subsurface(rect)
    pygame.image.save(sub, "image.jpg")
    win.fill((255, 255, 255))

    # Converting an image to an array
    img_array = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(img_array)
    image = cv2.resize(image, (28, 28))

    # Converting to a numpy array
    X = np.array(image)

    # Scaling data and shape
    X = X / 255.0
    X = tf.expand_dims(X, [0])
    X = tf.keras.utils.normalize(X, axis=1)

    # Getting predicted value and displaying on the screen
    prediction = model.predict(X)
    result_text = result_font.render(f"Prediction: {np.argmax(prediction)}", True, (0, 0, 0))
    win.blit(result_text, (width / 3, height / 2))


run = True
predicted = False
mouse_down = False

while run:
    # Mouse coordinates
    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_down = True
            if predicted:
                win.fill((255, 255, 255))
                predicted = False

        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_down = False

            # If clear button was pressed
            if (width - 55) > mouse[0] >= (width - 120) and 30 >= mouse[1] >= 0:
                win.fill((255, 255, 255))

            # If OK button was pressed
            elif mouse[0] >= (width - 55) and 30 >= mouse[1] >= 0:
                ok_pressed()
                predicted = True

    if mouse_down and mouse[1] >= 50:
        pygame.draw.circle(win, (0, 0, 0), mouse, 24)

    # Clear button
    if (width - 55) > mouse[0] >= (width - 120) and 30 >= mouse[1] >= 0:
        pygame.draw.rect(win, (220, 220, 220), [width - 120, 0, 60, 30])

    else:
        pygame.draw.rect(win, (250, 250, 250), [width - 120, 0, 60, 30])

    # OK button
    if mouse[0] >= (width - 55) and 30 >= mouse[1] >= 0:
        pygame.draw.rect(win, (220, 220, 220), [width - 55, 0, 70, 30])

    else:
        pygame.draw.rect(win, (250, 250, 250), [width - 55, 0, 70, 30])

    # Displaying text on buttons
    win.blit(ok_text, (width - 40, 10))
    win.blit(erase_text, (width - 110, 10))

    # Border line
    pygame.draw.line(win, (0, 0, 0), (0, 29), (width, 29))

    pygame.display.update()

pygame.quit()
