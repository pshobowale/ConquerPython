import numpy as np
import matplotlib.pyplot as plt
import pickle 



class GameMap:
    def __init__(self, fpath="Data/data.bin"):
        self._Pixel2Label,self._BackgroundMask,self._Label2Pixel,self._AdjacencyDict=pickle.load(open(fpath,"rb"))
        self._Background=np.zeros((self._BackgroundMask.shape[1],self._BackgroundMask.shape[0],3),dtype=np.uint8)
        self._Background[self._BackgroundMask.T]=[0,155,0]
        self._Background[self._BackgroundMask.T==0]=[100,100,255]

    def getMap(self):
        return self._Background
 
    def getLabel(self,x,y):
        return self._Label2Pixel[x,y]

    def getPixelMask(self,country_label):
        '''
            Get the mask and it's postion to colorize  a country

            Args:
                country_label (int): Label of the country

            Returns: (bool_array,(x,y))
        '''
        return self._Label2Pixel[country_label]

    def update():
        return self.getMap()