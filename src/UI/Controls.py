import pygame
from pygame.locals import *
import constants as const
from typing import Callable

class Controls:
    def __init__(self,click_callback:Callable[[int,int],None]):
        self.pan_box    = pygame.Rect( 0, 0, const.PAN_BOX_WIDTH, const.PAN_BOX_HEIGHT )  # curent pan "cursor position"
        self.pan_zoom   = 0

        self.mouse_movement=False
        self.selected_country=(-1,-1)
        self.click_callback=click_callback

        pygame.event.set_allowed([QUIT])
        pygame.event.set_allowed([FINGERDOWN,FINGERUP,FINGERMOTION])
        pygame.event.set_allowed([MOUSEBUTTONDOWN,MOUSEBUTTONUP,MOUSEWHEEL,KEYDOWN,KEYUP])

    def CheckTouch(self):
        # Handle user-input
        dx,dy,dz=0,0,0
        dx,dy=pygame.mouse.get_rel()
        click=False
        zoom=False
        finger_id=0
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
                print("FUP")
            
            elif  event.type==FINGERDOWN and not self.mouse_movement:
                print("FD")
            
            elif event.type==FINGERMOTION:
                if not self.mouse_movement:
                    self.mouse_movement=True
                #if event.finger_id>0:
                #    zoom=True
                #    click=False
                print("FM")
     
        if self.mouse_movement:
            self.pan_box.x-= const.PAN_VEL*dx/20
            self.pan_box.y-=const.PAN_VEL*dy/10

        if click:
            self.OnClick(pygame.mouse.get_pos())

            
        if zoom:
            dz=pygame.mouse.get_rel()[1]
            self.pan_zoom+=const.PAN_VEL/3000*dz/10
            

    def CheckMouse(self):
        # Handle user-input
        dx,dy,dz=0,0,0
        click=False
        pan_step=const.PAN_VEL

        for event in pygame.event.get():
            if event.type == QUIT:
               pygame.quit()
               return
            elif event.type == MOUSEWHEEL:
               dz+=pan_step*event.y
           
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
            self.pan_box.x+= pan_step*dx/10
            self.pan_box.y+=pan_step*dy/10
        else:
            click=True
            

        if click:
            self.OnClick(pygame.mouse.get_pos())

            

    def CheckKeyboard(self):

        #Movement keys
        #Pan-box moves up/down/left/right, Zooms with + and -
        keys = pygame.key.get_pressed()
        
        pan_step=const.PAN_VEL*self.pan_box.width/const.PAN_BOX_WIDTH_MAX

        if ( keys[pygame.K_UP] ):
            self.pan_box.y -= pan_step
        if ( keys[pygame.K_DOWN] ):
             self.pan_box.y += pan_step
        if ( keys[pygame.K_LEFT] ):
            self.pan_box.x -= pan_step
        if ( keys[pygame.K_RIGHT] ):
            self.pan_box.x += pan_step
        if ( keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS] ):
            self.pan_zoom+=const.PAN_VEL/3000
            
        if ( keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS] ):
            self.pan_zoom-=const.PAN_VEL/3000

    def CheckPanBoxMovement(self):
        # Ensure the pan-box stays within image
        if ( self.pan_box.x < 0 ):
            self.pan_box.x = 0 
        elif ( self.pan_box.x + self.pan_box.width >= const.GAMEMAP_WIDTH ):
            self.pan_box.x = const.GAMEMAP_WIDTH- self.pan_box.width
        if ( self.pan_box.y < 0 ):
            self.pan_box.y = 0 
        elif ( self.pan_box.y + self.pan_box.height >= const.GAMEMAP_HEIGHT ):
            self.pan_box.y = const.GAMEMAP_HEIGHT - self.pan_box.height

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
        self.CheckTouch()
        #self.CheckMouse()
        #self.CheckKeyboard()
        self.CheckPanBoxMovement()
        self.CheckPanBoxZoom()


    def OnClick(self,win_pos):
        wx,wy=win_pos[0]/const.WINDOW_WIDTH,win_pos[1]/const.WINDOW_HEIGHT
        x0,y0,dx,dy=self.pan_box
        px,py= round(x0+dx*wx),round(y0+dy*wy)
        self.click_callback(px,py)

    def Update(self)->pygame.Rect:
        self.CheckControls()
        return self.pan_box