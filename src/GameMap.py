import numpy as np
import matplotlib.pyplot as plt
import pickle 



class GameMap:
    def __init__(self, fpath="../Data/data.bin"):
        self._Pixel2Label,self._BackgroundMask,self._Label2Pixel,self._AdjacencyDict=pickle.load(open(fpath,"rb"))
        self._Background=np.zeros((self._BackgroundMask.shape[1],self._BackgroundMask.shape[0],3),dtype=np.uint8)
        self._Background[self._BackgroundMask.T]=[0,155,0]
        self._Background[self._BackgroundMask.T==0]=[100,100,255]

    def getMap(self):
        return self._Background
 
    def getLabel(self,x,y):
        return self._Pixel2Label[y,x]

    def getPixelMaskByLabel(self,country_label):
        '''
            Get the mask and it's postion to colorize  a country

            Args:
                country_label (int): Label of the country

            Returns: (bool_array)
        '''
        if len(self._Label2Pixel[country_label])!=0:
            mask=np.zeros(self._Background.shape[:2],dtype=bool)
            x,y=self._Label2Pixel[country_label][0][1]
            w,h=self._Label2Pixel[country_label][0][0].shape
            mask[y:y+h,x:x+w]=self._Label2Pixel[country_label][0][0].transpose()
            print(country_label)
            return mask
        else:
            return None

    def getPixelMaskByPos(self,x,y):
        '''
            Get the mask and it's postion to colorize  a country

            Args:
                x (int): X-Position in Image
                y (int): Y-Position in Image

            Returns: (bool_array)
        '''
        return self.getPixelMaskByLabel(self._Pixel2Label[y,x])
    def update():
        return self.getMap()
