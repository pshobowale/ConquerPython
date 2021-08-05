import pygame
from pygame.locals import *
import sys
import GameMap
import constants as const
import numpy as np
from UI.Controls import Controls
from typing import Union

class Engine:

    def __init__(self,Map):
        self.Map=Map
        # Window size

        ### PyGame initialisation
        pygame.init()
        self.window = pygame.display.set_mode( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ),
                                                 pygame.HWSURFACE|pygame.DOUBLEBUF)
        pygame.display.set_caption("Conquer")
        self.current_map= pygame.pixelcopy.make_surface(self.Map.getMap())
        
        ### Pan-position
        self.background = pygame.Surface( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ) )   # zoomed section is copied here
        self.zoom_image = None
        self.last_box   = pygame.Rect( 0, 0, 1, 1 )
        self.Controls   = Controls(self.ColorCountryByPos)
        

        ### Main Loop
        self.clock = pygame.time.Clock()
        self.done = False

        country_list=[c for c in range(1,Map.getNumCountrys())]
        self.ColorCountryByID(country_list,color=const.COLOR_COUNTRY,update_ui=False)
        self.base_image=self.current_map.copy()


    def UpdateUI(self,force_refresh=False):
        pan_box=self.Controls.Update()
        if ( pan_box != self.last_box ) or force_refresh:
            # Create a new sub-image but only if the size changed
            # otherwise we can just re-use it
            if ( pan_box.width != self.last_box.width or pan_box.height != self.last_box.height ):
                self.zoom_image = pygame.Surface( ( pan_box.width, pan_box.height ) )  
        
            self.zoom_image.blit( self.current_map, ( 0, 0 ), pan_box )                  # copy base image
            #print(self.pan_box.height,self.pan_box.width)
            pygame.transform.scale( self.zoom_image, const.WINDOW_SIZE, self.background )     # scale into thebackground
            self.last_box = pan_box.copy()                                         # copy current position

        self.window.blit(self.background, ( 0, 0 ) )
        pygame.display.flip()
        self.clock.tick_busy_loop(60)


    def ColorCountryByPos(self,px,py,color=const.COLOR_COUNTRY_SEL,uncolor_previsous=True,update_ui=True):
        if uncolor_previsous:
            cmap=pygame.surfarray.array3d(self.base_image)
        else:
            cmap=pygame.surfarray.array3d(self.current_map)
        
        mask=self.Map.getPixelMaskByPos(px,py)
        if mask is not None:
            cmap[mask==True]=color
            pygame.surfarray.blit_array(self.current_map,cmap)
            if update_ui:
                self.UpdateUI(True)
    
    def ColorCountryByID(self,country_id:Union[list[int],int],color=const.COLOR_COUNTRY_SEL,uncolor_previsous=False,update_ui=True):
        if uncolor_previsous:
            cmap=pygame.surfarray.array3d(self.base_image)
        else:
            cmap=pygame.surfarray.array3d(self.current_map)

        if type(country_id) is int:
            country_id=[country_id]

        mask=np.zeros(const.GAMEMAP_SIZE,dtype=bool)
        for cid in country_id:
            cmask=self.Map.getPixelMaskByID(cid)
            if cmask is not None:
                mask=cmask|mask

        if mask is not None:
            cmap[mask==True]=color
            pygame.surfarray.blit_array(self.current_map,cmap)
            if update_ui:
                self.UpdateUI(True)

