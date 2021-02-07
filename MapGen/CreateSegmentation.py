
import numpy as np
import matplotlib.pyplot as plt
import pickle
import lab  
from datetime import datetime


def create_seg(fname="./img/Europe.png",debug=False):
    c=plt.imread(fname)[:,1150:,:3]
    c[-100:,2400:4000]=[1,1,1]
    c=c*255
    #img=c[2000:3000,2000:3000]
    img=c

    if debug:   
        plt.imshow(np.array(img,dtype=np.uint8))
        plt.title("Original")
        plt.show()

    S=lab.region_growing_rgb(img, 1, min_pixels=500, smoothing=0)
    print("Found", np.max(S)," different regions")
    dt_string = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
    pickle.dump( S, open( "./2big4git/SegMap-"+dt_string+".seg", "wb" ) )
    print("Map generated at ",dt_string)

    if debug:
        I= lab.segments_to_image(S)
        plt.imshow(np.array(I,dtype=np.uint8))
        plt.title("Segmentation")
        plt.show()

    return S

if __name__ == "__main__":
    create_seg(debug=False)