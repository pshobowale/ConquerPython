import pygame
from pygame.locals import *
import sys
import GameMap
import constants as const
import numpy as np
from UI.Controls import Controls
from typing import Union
import time
import matplotlib.pyplot as plt

class Engine:

    def __init__(self,Map):
        self.Map=Map
        # Window size
        self.update_counter=0
        ### PyGame initialisation
        pygame.init()
        self.window = pygame.display.set_mode( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ),
                                                 pygame.HWSURFACE|pygame.DOUBLEBUF)
        pygame.display.set_caption("Conquer")
        self.current_map= pygame.pixelcopy.make_surface(self.Map.getMap()).convert()
        
        ### Pan-position
        self.background = pygame.Surface( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ) )   # zoomed section is copied here
        self.zoom_image = None
        self.last_box   = pygame.Rect( 0, 0, 1, 1 )
        self.Controls   = Controls(self.ColorCountryByPos)
        

        ### Main Loop
        self.clock = pygame.time.Clock()
        self.done = False

        if self.Map.GameMapLoaded is False:
            country_list=[c for c in range(1,Map.getNumCountrys())]
            self.ColorCountryByID(country_list,color=const.COLOR_COUNTRY)
        self.base_image=self.current_map.copy()
        self.UpdateUI(force_refresh=True)

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
            print("update",self.update_counter,pan_box,force_refresh)
            self.last_box = pan_box.copy()                                         # copy current position

            self.window.blit(self.background, ( 0, 0 ) )
            pygame.display.flip()
            
            self.update_counter+=1
        self.clock.tick_busy_loop(60)


    def ColorCountryByPos(self,px,py,color=const.COLOR_COUNTRY_SEL,uncolor_previsous=True):
        t0=time.time_ns()
        color=pygame.Color(color)

        if uncolor_previsous:
            self.current_map=self.base_image.copy()
            
        cmap=pygame.PixelArray(self.current_map)
        t1=time.time_ns()
        mask=self.Map.getPixelMaskByPos(px,py)
        t2=time.time_ns()
        if mask is not None:
            (x0,y0,w,h),mask=mask
            for i,j in zip(*np.where(mask)):
                j,i= int(i),int(j)
                cmap[y0+j,x0+i] = color
            
        cmap.close()

        self.UpdateUI(force_refresh=True)
        t3=time.time_ns()
        
        total=(t3-t0)/100
        print("Total: ",total// 10_000)
        print("Surfarray Conversion: ",(t1-t0)/total)
        print("Mask Creation: ",(t2-t1)/total )
        print("Update: ",(t3-t2)/total)
        
    
    def ColorCountryByID(self,country_id:Union[list[int],int],color=const.COLOR_COUNTRY_SEL,uncolor_previsous=False):
        t0=time.time_ns()
        if uncolor_previsous:
            cmap=pygame.surfarray.array3d(self.base_image)
        else:
            cmap=pygame.surfarray.array3d(self.current_map)
        t1=time.time_ns()
        if type(country_id) is int:
            country_id=[country_id]

        mask=np.zeros(const.GAMEMAP_SIZE,dtype=bool)
        for cid in country_id:
            cmask=self.Map.getPixelMaskByID(cid)
            if cmask is not None:
                mask=cmask|mask
        t2=time.time_ns()
        if mask is not None:
            cmap[mask==True]=color
            pygame.surfarray.blit_array(self.current_map,cmap)
            self.UpdateUI(force_refresh=True)
        t3=time.time_ns()

        total=(t3-t0)/100
        print("Total: ",total// 10_000)
        print("Surfarray Conversion: ",(t1-t0)/total)
        print("Mask Creation: ",(t2-t1)/total )
        print("Update: ",(t3-t2)/total)

 