import pygame
import sys
import GameMap
import UI

Map=GameMap.GameMap()
UI=UI.Engine(Map)

clock = pygame.time.Clock()
done = False


while not UI.done:

    UI.CheckControls()
    UI.UpdateUI()
    UI.clock.tick_busy_loop(60)


pygame.quit()