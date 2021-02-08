import pygame
from pygame.locals import *
import sys
import GameMap

class Engine:

    def __init__(self,Map):
        self.Map=Map
        # Window size
        self.WINDOW_WIDTH    = 800
        self.WINDOW_HEIGHT   = 400
        self.WINDOW_SURFACE  = pygame.HWSURFACE|pygame.DOUBLEBUF|pygame.RESIZABLE

        self.PAN_BOX_WIDTH_MAX   = 4000
        self.PAN_BOX_HEIGHT_MAX  = 2000

        self.PAN_BOX_WIDTH   = 4000
        self.PAN_BOX_HEIGHT  = 2000
        self.PAN_STEP        = 100

        ### PyGame initialisation
        pygame.init()
        self.window = pygame.display.set_mode( ( self.WINDOW_WIDTH, self.WINDOW_HEIGHT ), self.WINDOW_SURFACE )
        pygame.display.set_caption("Conquer")
        self.base_image = pygame.pixelcopy.make_surface(self.Map.getMap())
        
        ### Pan-position
        self.background = pygame.Surface( ( self.WINDOW_WIDTH, self.WINDOW_HEIGHT ) )   # zoomed section is copied here
        self.zoom_image = None
        self.pan_box    = pygame.Rect( 0, 0, self.PAN_BOX_WIDTH, self.PAN_BOX_HEIGHT )  # curent pan "cursor position"
        self.last_box   = pygame.Rect( 0, 0, 1, 1 )

        ### Mouse Movement
        self.mouse_movement=False
        self.selected_country=(-1,-1)

        ### Main Loop
        self.clock = pygame.time.Clock()
        self.done = False


    def CheckMouse(self):
        # Handle user-input
        dx,dy,dz=0,0,0
        click=False
        for event in pygame.event.get():
            if event.type == QUIT:
               pygame.quit()
               return
            elif event.type == MOUSEWHEEL:
               dz+=self.PAN_STEP*event.y
           
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
            self.pan_box.x+= self.PAN_STEP*dx/10
            self.pan_box.y+=self.PAN_STEP*dy/10
        else:
            click=True

        if click:
            print("Click", pygame.mouse.get_pos())


    def CheckKeyboard(self):

        #Movement keys
        #Pan-box moves up/down/left/right, Zooms with + and -
        keys = pygame.key.get_pressed()
        
        if ( keys[pygame.K_UP] ):
            self.pan_box.y -= self.PAN_STEP
        if ( keys[pygame.K_DOWN] ):
             self.pan_box.y += self.PAN_STEP
        if ( keys[pygame.K_LEFT] ):
            self.pan_box.x -= self.PAN_STEP
        if ( keys[pygame.K_RIGHT] ):
            self.pan_box.x += self.PAN_STEP
        if ( keys[pygame.K_PLUS] or keys[pygame.K_EQUALS] ):
            self.pan_box.width  += self.PAN_STEP*2
            self.pan_box.height += self.PAN_STEP
            if ( self.pan_box.width > self.PAN_BOX_WIDTH_MAX):  # Ensure size is sane
                self.pan_box.width = self.PAN_BOX_WIDTH_MAX
            if ( self.pan_box.height > self.PAN_BOX_HEIGHT_MAX):
                self.pan_box.height = self.PAN_BOX_HEIGHT_MAX
        if ( keys[pygame.K_MINUS] ):
            self.pan_box.width  -= self.PAN_STEP*2
            self.pan_box.height -= self.PAN_STEP
            if ( self.pan_box.width < self.PAN_STEP ):  # Ensure size is sane
                self.pan_box.width = self.PAN_STEP
            if ( self.pan_box.height < self.PAN_STEP ):
                self.pan_box.height = self.PAN_STEP

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
        if ( self.pan_box != self.last_box ):
            # Create a new sub-image but only if the size changed
            # otherwise we can just re-use it
            if ( self.pan_box.width != self.last_box.width or self.pan_box.height != self.last_box.height ):
                self.zoom_image = pygame.Surface( ( self.pan_box.width, self.pan_box.height ) )  
        
            self.zoom_image.blit( self.base_image, ( 0, 0 ), self.pan_box )                  # copy base image
            window_size = ( self.WINDOW_WIDTH, self.WINDOW_HEIGHT )
            pygame.transform.scale( self.zoom_image, window_size, self.background )     # scale into thebackground
            self.last_box = self.pan_box.copy()                                         # copy current position

        self.window.blit(self.background, ( 0, 0 ) )
        pygame.display.flip()