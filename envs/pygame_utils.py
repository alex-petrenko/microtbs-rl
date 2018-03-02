"""
Pygame helpers.

"""


import pygame


def get_events(block=False):
    while True:
        events = pygame.event.get()
        if not block:
            return events
        for event in events:
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                return events
