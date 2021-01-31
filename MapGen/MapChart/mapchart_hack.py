for i in range(0xff0000,0xffffff):
    print("\"{:x}\"".format(i),end=",")
    if i%15==0:
        print()