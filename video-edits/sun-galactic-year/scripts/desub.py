import cv2, numpy as np
Y0,Y1,X0,X1=1440,1560,40,1040
K=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17))
K2=cv2.getStructuringElement(cv2.MORPH_RECT,(25,9))
def submask(band):
    b,g,r=[band[...,i].astype(np.int16) for i in range(3)]
    mn=np.minimum(np.minimum(b,g),r); mx=np.maximum(np.maximum(b,g),r)
    white=(mn>175)&(mx-mn<60)
    red=(r>150)&(g<100)&(b>35)&(b<160)&(r-g>90)
    m=((white|red)*255).astype(np.uint8)
    # keep only components that look like glyphs: height 20-75 px, and sitting in the text row
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    keep=np.zeros_like(m)
    for k in range(1,n):
        x,y,w,h,a=st[k]
        if 12<=h<=80 and w<=160 and a>=25: keep[lab==k]=255
    # require a text-like cluster: many glyphs on same row
    if keep.sum()/255<300: return None
    # close gaps between letters, then grow over the black stroke + drop shadow
    m=cv2.dilate(cv2.morphologyEx(keep,cv2.MORPH_CLOSE,K2),K,iterations=1)
    return m
def clean(frame):
    band=frame[Y0:Y1,X0:X1]
    m=submask(band)
    if m is not None:
        frame[Y0:Y1,X0:X1]=cv2.inpaint(band,m,9,cv2.INPAINT_TELEA)
    return frame
if __name__=="__main__":
    import sys
    cap=cv2.VideoCapture(sys.argv[1])
    for t in [1.2,4.4,9.0,24.0,26.4,31.0,42.6,46.0,49.6]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,f=cap.read()
        a=f[1380:1620].copy(); c=clean(f)[1380:1620]
        cv2.imwrite(f"{sys.argv[2]}/t{t}.jpg",cv2.resize(np.vstack([a,c]),(540,240)))
