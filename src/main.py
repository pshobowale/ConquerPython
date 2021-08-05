import pygame
import sys
import GameMap
import UI.Engine as UI

Map=GameMap.GameMap()
UI=UI.Engine(Map)

done = False


while not UI.done:
    UI.UpdateUI()
    


pygame.quit()