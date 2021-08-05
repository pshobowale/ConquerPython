import pygame
from pygame.locals import *
import sys
import GameMap
import constants as const
import numpy as np

class Engine:

    def __init__(self,Map):
        self.Map=Map
        # Window size

        ### PyGame initialisation
        pygame.init()
        self.window = pygame.display.set_mode( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ),
                                                 pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE)
        pygame.display.set_caption("Conquer")
        self.base_image = pygame.pixelcopy.make_surface(self.Map.getMap())
        self.current_map=self.base_image
        
        ### Pan-positionpx,py
        self.background = pygame.Surface( ( const.WINDOW_WIDTH, const.WINDOW_HEIGHT ) )   # zoomed section is copied here
        self.zoom_image = None
        self.pan_box    = pygame.Rect( 0, 0, const.PAN_BOX_WIDTH, const.PAN_BOX_HEIGHT )  # curent pan "cursor position"
        self.pan_zoom   = 0
        self.last_box   = pygame.Rect( 0, 0, 1, 1 )

        ### Mouse Movement
        self.mouse_movement=False
        self.selected_country=(-1,-1)

        ### Main Loop
        self.clock = pygame.time.Clock()
        self.done = False

    def CheckTouch(self):
        # Handle user-input
        dx,dy,dz=0,0,0
        dx,dy=pygame.mouse.get_rel()
        click=False
        zoom=False
        for event in pygame.event.get():
            if event.type == QUIT:
               pygame.quit()
               return
            elif event.type==FINGERUP:
                if not self.mouse_movement:
                    click=True
                else:
                    click=False
                self.mouse_movement=False 
                #print("FUP")
            
            #elif  event.type==FINGERDOWN and not self.mouse_movement:
            #    print("FD")
            
            elif event.type==FINGERMOTION:
                if not self.mouse_movement:
                    self.mouse_movement=True
                if event.finger_id>0:
                    zoom=True
                    click=False
                #print("FM")
     
        if self.mouse_movement:
            self.pan_box.x-= const.PAN_STEP*dx/10
            self.pan_box.y-=const.PAN_STEP*dy/10

        if click:
            self.CountrySelected(pygame.mouse.get_pos())

            
        if zoom:
            dz=pygame.mouse.get_rel()[1]
            self.pan_box.width  -= const.PAN_STEP*2*dz/10
            self.pan_box.height -= const.PAN_STEP*dz/10
            if ( self.pan_box.width < const.PAN_STEP ):  # Ensure size is sane
                self.pan_box.width = const.PAN_STEP
            if ( self.pan_box.height < const.PAN_STEP ):
                self.pan_box.height = const.PAN_STEP
            if ( self.pan_box.width > const.PAN_BOX_WIDTH_MAX):  # Ensure size is sane
                self.pan_box.width = const.PAN_BOX_WIDTH_MAX
            if ( self.pan_box.height > const.PAN_BOX_HEIGHT_MAX):
                self.pan_box.height = const.PAN_BOX_HEIGHT_MAX

    def CheckMouse(self):
        # Handle user-input
        dx,dy,dz=0,0,0
        click=False
        for event in pygame.event.get():
            if event.type == QUIT:
               pygame.quit()
               return
            elif event.type == MOUSEWHEEL:
               dz+=const.PAN_STEP*event.y
           
            elif event.type==MOUSEBUTTONUP:
                if not self.mouse_movement:
                    click=True
                self.mouse_movement=False   

            elif  event.type==MOUSEBUTTONDOWN or self.mouse_movement:
                dx,dy=pygame.mouse.get_rel()
                if(dx or dy):
                    self.mouse_movement=True
            
                    

            
        dmax=20
        if(dx<dmax and dy<dmax and dx>-dmax and dy>-dmax):
            self.pan_box.x+= const.PAN_STEP*dx/10
            self.pan_box.y+=const.PAN_STEP*dy/10
        else:
            click=True

        if click:
            self.CountrySelected(pygame.mouse.get_pos())
            

    def CheckKeyboard(self):

        #Movement keys
        #Pan-box moves up/down/left/right, Zooms with + and -
        keys = pygame.key.get_pressed()
        
        if ( keys[pygame.K_UP] ):
            self.pan_box.y -= const.PAN_STEP
        if ( keys[pygame.K_DOWN] ):
             self.pan_box.y += const.PAN_STEP
        if ( keys[pygame.K_LEFT] ):
            self.pan_box.x -= const.PAN_STEP
        if ( keys[pygame.K_RIGHT] ):
            self.pan_box.x += const.PAN_STEP
        if ( keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS] ):
            self.pan_zoom+=const.PAN_STEP/3000
            
        if ( keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS] ):
            self.pan_zoom-=const.PAN_STEP/3000

    def CheckPanBoxMovement(self):
        # Ensure the pan-box stays within image
        if ( self.pan_box.x < 0 ):
            self.pan_box.x = 0 
        elif ( self.pan_box.x + self.pan_box.width >= self.base_image.get_width() ):
            self.pan_box.x = self.base_image.get_width() - self.pan_box.width - 1
        if ( self.pan_box.y < 0 ):
            self.pan_box.y = 0 
        elif ( self.pan_box.y + self.pan_box.height >= self.base_image.get_height() ):
            self.pan_box.y = self.base_image.get_height() - self.pan_box.height - 1

    def CheckPanBoxZoom(self):
        if(self.pan_zoom!=0):
            self.pan_box.width*=2**self.pan_zoom
            self.pan_box.height*=2**self.pan_zoom
            self.pan_zoom=0

            if self.pan_box.width>const.PAN_BOX_WIDTH_MAX or self.pan_box.height>const.PAN_BOX_HEIGHT_MAX:
                self.pan_box.width=const.PAN_BOX_WIDTH_MAX
                self.pan_box.height=const.PAN_BOX_HEIGHT_MAX

            if self.pan_box.width<const.PAN_BOX_WIDTH_MIN or self.pan_box.height<const.PAN_BOX_HEIGHT_MIN:
                self.pan_box.width=const.PAN_BOX_WIDTH_MIN
                self.pan_box.height=const.PAN_BOX_HEIGHT_MIN

    def CheckControls(self):
        self.CheckMouse()
        self.CheckKeyboard()
        self.CheckPanBoxMovement()
        self.CheckPanBoxZoom()

    def UpdateUI(self,force_refresh=False):
        if ( self.pan_box != self.last_box ) or force_refresh:
            # Create a new sub-image but only if the size changed
            # otherwise we can just re-use it
            if ( self.pan_box.width != self.last_box.width or self.pan_box.height != self.last_box.height ):
                self.zoom_image = pygame.Surface( ( self.pan_box.width, self.pan_box.height ) )  
        
            self.zoom_image.blit( self.current_map, ( 0, 0 ), self.pan_box )                  # copy base image
            #print(self.pan_box.height,self.pan_box.width)
            pygame.transform.scale( self.zoom_image, const.WINDOW_SIZE, self.background )     # scale into thebackground
            self.last_box = self.pan_box.copy()                                         # copy current position

        self.window.blit(self.background, ( 0, 0 ) )
        pygame.display.flip()

    def CountrySelected(self,win_pos):
        x0,y0,dx,dy=self.pan_box
        wx,wy=win_pos[0]/const.WINDOW_WIDTH,win_pos[1]/const.WINDOW_HEIGHT
        px,py= round(x0+dx*wx),round(y0+dy*wy)
        #print(self.pan_box)
        #print(wx,wy)
        #print(px,py)
        self.ColorCountry(px,py)
        
    def ColorCountry(self,px,py):
        if self.Map.getLabel(px,py)==0:
            return
        print(px,py)
        self.current_map=self.base_image
        cmap=pygame.surfarray.array2d(self.base_image)
        np.set_printoptions(formatter={'int':hex})
        
        mask=self.Map.getPixelMaskByPos(px,py)
        if mask is not None:
            cmap[mask==True]=0
            pygame.surfarray.blit_array(self.current_map,cmap)
            self.UpdateUI(True)
            