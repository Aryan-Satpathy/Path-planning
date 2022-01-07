from os import close
import cv2
import numpy as np
import math
import time

SHOW_EXPLORATION = True

fps = 144

StartColor = np.array((113, 204, 45), dtype = np.uint8)
ObstacleColor = np.array((255, 255, 255), dtype = np.uint8)
EndColor = np.array((60, 76, 231), dtype = np.uint8)

'''
StartColor = np.array((255, 0, 0), dtype = np.uint8)
ObstacleColor = np.array((0, 0, 0), dtype = np.uint8)
EndColor = np.array((0, 255, 0), dtype = np.uint8)
'''

Start = (0, 0)
Destination = (99, 99)

'''
class cell :
    def __init__(self, x, y, color) :
        global Start
        global Destination
        self.pos = (x, y)
        self.isStart = False not in (StartColor == color)
        self.isDestination = False not in (EndColor == color)
        self.isNavigable = False not in (ObstacleColor != color)
        self.f = [math.inf, 0][self.isStart]
        self.g = 0
        if self.isStart : Start = self.pos; openList.append((x, y))
        if self.isDestination : Destination = self.pos
        self.successor = None
'''

# Resize
# frame = np.zeros((1000, 1000, 3), dtype = np.uint8)
def resize(img, factor = 10) : 
    frame = np.ones((factor * img.shape[0], factor * img.shape[1], 3), dtype = np.uint8)

    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            frame[factor * i : factor * i + factor, factor * j : factor * j + factor] *= img[i][j]

    return frame

import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_path = "../Images/Task_1_Low.png"
path = os.path.join(script_dir, rel_path)
# path = r'..\Images\Task_1_Low.png'
# path = r'C:\Users\ARYAN SATPATHY\Downloads\pixil-frame-0 (1).png'
img = cv2.imread(path)

# cv2.imshow('kjfjbjkbw', resize(img))
# cv2.waitKey(0)

# cv2.destroyAllWindows()

# print(img is None)

# print(img.shape)

# Path Detection here
def reconstruct_path(cameFrom, current) :
    total_path = [current]
    while current in cameFrom:
        current = cameFrom[current]
        total_path.append(current)
    return total_path[ : : -1]

def BFS(start, goal, navigable, img, costUpdation = False) : 
    openSetPos = [start]
    openSetCost = [0]
    closedList = {}
    cameFrom = {}
    lastUpdate = time.time()

    while len(openSetPos) : 
        curr, currcost = openSetPos[0], openSetCost[0]
        img[curr[0], curr[1]] = np.array([100, 255, 100])
        if SHOW_EXPLORATION and (time.time() - lastUpdate) >= 1 / fps :
            frame = resize(img)
            cv2.imshow('Path plannning', frame)
            cv2.waitKey(1)
            lastUpdate = time.time()
        if curr == goal : 
            if costUpdation : 
                pass
            else : 
                if SHOW_EXPLORATION :
                    frame = resize(img)
                    cv2.imshow('Path plannning', frame)
                    cv2.waitKey(1)
                    lastUpdate = time.time()
                return reconstruct_path(cameFrom, curr)
        
        neighbours = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]

        openSetPos.pop(0)
        openSetCost.pop(0)
        if closedList.get(curr, math.inf) > currcost : 
            closedList[curr] = currcost

        for n in neighbours : 
            tentativecost = currcost + [1, 1.414][n[0] * n[1]]
            neighbour = curr[0] + n[0], curr[1] + n[1]
            if 0 > neighbour[0] or neighbour[0] >= 100 or 0 > neighbour[1] or neighbour[1] >= 100 or not navigable[neighbour[0]][neighbour[1]] : 
                continue
            if neighbour not in openSetPos and neighbour not in closedList : 
                openSetPos.append(neighbour)
                openSetCost.append(tentativecost)
                cameFrom[neighbour] = curr
            elif neighbour not in openSetPos :
                var = closedList.get(neighbour, None)
                if var == None : 
                    pass
                elif var != None and var > tentativecost : 
                    closedList[neighbour] = tentativecost
                    cameFrom[neighbour] = curr
            elif neighbour not in closedList : 
                ind = openSetPos.index(neighbour)
                if openSetCost[ind] > tentativecost : 
                    openSetCost[ind] = tentativecost
                    cameFrom[neighbour] = curr 
    if costUpdation : 
        return reconstruct_path(cameFrom, goal)
# A* finds a path from start to goal.
# h is the heuristic function. h(n) estimates the cost to reach goal from node n.
def A_Star(start, goal, h, navigable, img) : 
    # The set of discovered nodes that may need to be (re-)expanded.
    # Initially, only the start node is known.
    # This is usually implemented as a min-heap or priority queue rather than a hash-set. 
    
    openSet = [start]

    # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from start
    # to n currently known.
    cameFrom = {}

    # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
    gScore = [[math.inf for i in range(100)] for j in range(100)]
    gScore[start[0]][start[1]] = 0

    # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
    # how short a path from start to finish can be if it goes through n.
    fScore = [[math.inf for i in range(100)] for j in range(100)]
    fScore[start[0]][start[1]] = h(start, goal)

    while len(openSet) :
        # print(len(openSet))
        # This operation can occur in O(1) time if openSet is a min-heap or a priority queue
        current = min(openSet, key = lambda pos : fScore[pos[0]][pos[1]])                                       # Get node from openSet with min fScore
        if current == goal :            
            print("Cost = {}".format(gScore[current[0]][current[1]] + [1, 1.4][x * y]))
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        # neighbours = [(-1, -1), ]
        neighbours = [(x, y) for x in range(-1, 2) for y in range(-1, 2) if (not(x == 0 and y == 0)) and 0 <= current[0] + x < 100 and 0 <= current[1] + y < 100 and navigable[current[0] + x][current[1] + y]]
        for neighbor in neighbours :
            # d(current,neighbor) is the weight of the edge from current to neighbor
            # tentative_gScore is the distance from start to the neighbor through current
            x, y = neighbor
            tentative_gScore = gScore[current[0]][current[1]] + [1, 1.4][x * y]
            if tentative_gScore < gScore[current[0] + x][current[1] + y] : 
                # This path to neighbor is better than any previous one. Record it!
                cameFrom[(current[0] + x, current[1] + y)] = current
                gScore[current[0] + x][current[1] + y] = tentative_gScore
                fScore[current[0] + x][current[1] + y] = gScore[current[0] + x][current[1] + y] + h((current[0] + x, current[1] + y), goal)
                if (current[0] + x, current[1] + y) not in openSet : 
                    openSet.append((current[0] + x, current[1] + y))
                    # img[current[0] + x, current[1] + y] = np.array([0, 0, 255])
                    # frame = resize(img)
                    # cv2.imshow('Path plannning', frame)
                    # cv2.waitKey(1)

    # Open set is empty but goal was never reached
    return None

def h_Manhattan(pos, goal) :
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def h_Diagonal(pos, goal) :
    delx, dely = abs(pos[0] - goal[0]), abs(pos[1] - goal[1])
    return (delx + dely) + (math.sqrt(2) - 2) * min(delx, dely)

def h_Euclidean(pos, goal) :
    return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)


navigable = [[False for y in range(100)] for x in range(100)]

for x in range(100) :
    for y in range(100) :
        if False not in (img[(x,  y)] == StartColor) :
            Start = (x, y)
        if False not in (img[(x,  y)] == EndColor) :
            Destination = (x, y)
            navigable[x][y] = True
        if False in (img[(x,  y)] == ObstacleColor) :
            navigable[x][y] = True

print('Destination = ', Destination)

_nav = [[0 for y in range(100)] for x in range(100)]
for x in range(100) : 
    for y in range(100) : 
        if navigable[x][y] : _nav[x][y] = 1

Frame = np.array(_nav, dtype= np.uint8)
Frame = Frame * 255

# cv2.imshow('Navigable', Frame)
# cv2.waitKey(0)
# cv2.destroyWindow('Navigable')

img1, img2, img3, img4 = img.copy(), img.copy(), img.copy(), img.copy()

# start = time.time()
# DiagonalPath = A_Star(Start, Destination, h_Diagonal, navigable, img1)
# end = time.time()
# print(end - start)
# cv2.waitKey(0)
# start = time.time()
# EuclideanPath = A_Star(Start, Destination, h_Euclidean, navigable, img2)
# end = time.time()
# print(end - start)
# cv2.waitKey(0)
# start = time.time()
# ManhattanPath = A_Star(Start, Destination, h_Manhattan, navigable, img3)
# end = time.time()
# print(end - start)
start = time.time()
BFSPath = BFS(Start, Destination, navigable, img4, costUpdation = True)
end = time.time()
print(end - start)
cv2.waitKey(0)
cv2.destroyAllWindows()

def Addpath(img, path, color) :
    for pixel in path :
        if pixel != Start and pixel != Destination : img[pixel] = np.array(color, dtype = np.uint8)

ManhattanColor = (235, 206, 135)                                                                    # Sky Blue
DiagonalColor = (0, 165, 255)                                                                       # Orange
EuclideanColor = (0, 255, 255)                                                                      # Yellow
BFSColor = (100, 100, 255)

img1, img2, img3, img4 = img.copy(), img.copy(), img.copy(), img.copy()

# Addpath(img1, ManhattanPath, ManhattanColor)
# Addpath(img2, DiagonalPath, DiagonalColor)
# Addpath(img3, EuclideanPath, EuclideanColor)
Addpath(img4, BFSPath, BFSColor)

# frame1 = resize(img1, 7)
# frame2 = resize(img2, 7)
# frame3 = resize(img3, 7)
frame4 = resize(img4, 7)

# img = cv2.resize(img, None, fx = 10, fy = 10, interpolation = cv2.INTER_CUBIC)

# cv2.imshow('Manhattan', frame1)
# cv2.imshow('Diagonal', frame2)
# cv2.imshow('Euclidean', frame3)
cv2.imshow('BFS', frame4)

key = cv2.waitKey(0)

cv2.destroyAllWindows()
