import numpy as np

def CompletePath(s, w, h) -> str:
    '''This function is used to escape from a room whose size is w * h.
    You are trapped in the bottom-left corner and need to cross to the
    door in the upper-right corner to escape.
    @:param s: a string representing the partial path composed of {'U', 'D', 'L', 'R', '?'}
    @:param w: an integer representing the room width
    @:param h: an integer representing the room length
    @:return path: a string that represents the completed path, with all question marks in with the correct directions
    or None if there is no path possible.
    '''
    
    # # TODO # #
    
    #error-check argument#
    assert isinstance(s,str), "s is a string"
    assert isinstance(w,int), "w is an int"
    assert isinstance(h,int), "h is an int"
    
    #Algorithmn starts here#
    dir = {'U':(0,1),'D':(0,-1),'L':(-1,0),'R':(1,0),"?":(0,0)}
    next = "UDLR"
    start = (1,1) # the start point
    goal = (w,h) # the goal point
    x,y = start # the current point
    mark = np.zeros((w+1,h+1)) #mark the passed point
    mark[1][1] = 1 #mark the start point
    idx = 0  # mark the index of the string
    for i in s:
        dx,dy=dir[i]
        if not(dx==0 and dy==0): #process the path if there is no ?

            #current point's coordinate
            x+=dx 
            y+=dy

            #is the point out of range?
            if x<1 or x>w or y<1 or y>h:
                return
            else:
                mark[x][y]=1#mark the current point

            #is the point reach the goal?
            if x==w and y==h :
                if idx == len(s)-1:
                    return s
                else:
                    return

            if idx==len(s)-1:
                return
                
        else: #process the path if there is ?

            # is the point reach the goal?
            if x==w and y==h:
                return s

            #enumerate 4 directions
            for j in next:
                dx,dy=dir[j]
                
                # next point's coordinate
                next_x=x+dx
                next_y=y+dy
                
                # is next point out of the range?
                if next_x<1 or next_x>w or next_y<1 or next_y>h:
                    continue
                
                # is next point marked?
                if mark[next_x][next_y] ==0:
                    mark[next_x][next_y] =1 # mark the next point
                    s = s[:idx]+j+s[idx+1:] #update the new string
                    new_path = CompletePath(s,w,h) #try next point
                    mark[next_x][next_y] =0 #unmark the next point
                    if new_path:
                        return new_path
                    else:
                        continue
                        
        idx+=1 # index increase
    pass

s=CompletePath("??????", 3, 3)
print(s)