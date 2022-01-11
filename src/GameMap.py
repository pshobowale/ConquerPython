import numpy as np
import matplotlib.pyplot as plt
import pickle 
import constants as const
from typing import Optional



class GameMap:
    def __init__(self, fpath:str=const.FILE_DATA_BIN,gm_path:str=const.FILE_GAMEMAP):
        self.GameMapLoaded=False
        
        self._Pixel2Label,self._BackgroundMask,self._Label2Pixel,self._AdjacencyDict=pickle.load(open(fpath,"rb"))
        self._Background=np.zeros((self._BackgroundMask.shape[1],self._BackgroundMask.shape[0],3),dtype=np.uint8)
        
        if gm_path is None:
            self._Background[self._BackgroundMask.T]= const.COLOR_BORDER
            self._Background[self._BackgroundMask.T==0]= const.COLOR_SEA
        else:
            self.GameMapLoaded=True
            self._Background=plt.imread(gm_path)[:,:,:3]*255
            self._Background=self._Background.astype(int)
        
    def getNumCountrys(self)-> int:
        return len(self._Label2Pixel)

    def getMap(self)->np.ndarray:
        return self._Background
 
    def getLabel(self,x,y)->int:
        return self._Pixel2Label[y,x]

    def getPixelMaskByID(self,country_id:int)-> Optional[np.ndarray]:
        '''
            Get the mask and it's postion to colorize  a country

            Args:
                country_label (int): Label of the country

            Returns: (bool_array)
        '''
        if len(self._Label2Pixel[country_id])!=0:
            mask=np.zeros(const.GAMEMAP_SIZE,dtype=bool)
            x,y=self._Label2Pixel[country_id][0][1]
            w,h=self._Label2Pixel[country_id][0][0].shape
            mask[y:y+h,x:x+w]=self._Label2Pixel[country_id][0][0].transpose()
            return mask
        else:
            return None

    def getPixelMaskByPos(self,x:int,y:int)->Optional[np.ndarray]:
        '''
            Get the mask and it's postion to colorize  a country

            Args:
                x (int): X-Position in Image
                y (int): Y-Position in Image

            Returns: ((x0,y0,w,h),bool_array)
        '''
        country_id=self._Pixel2Label[y,x]

        if len(self._Label2Pixel[country_id])!=0:
            x,y=self._Label2Pixel[country_id][0][1]
            w,h=self._Label2Pixel[country_id][0][0].shape
            mask=self._Label2Pixel[country_id][0][0].transpose()
            return ((x,y,w,h),mask)
        else:
            return None

