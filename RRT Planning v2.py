## Key difference between RRT Planning and RRT Planning v2 is 
## I made it so that all the nodes are connected to each other and shortest path is recalculated every now and then.
## It is way slower than normal RRT but produces quite accurate results.
## Might as well be more optimal than A star in some cases.

import cv2
import numpy as np
import math
import time
import random as rnd

def resize(img, factor = 10) : 
    frame = np.ones((factor * img.shape[0], factor * img.shape[1], 3), dtype = np.uint8)

    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            frame[factor * i : factor * i + factor, factor * j : factor * j + factor] *= img[i][j]

    return frame

# path = r'C:\Users\ARYAN SATPATHY\Downloads\Task_1_Low.png'
path = r'C:\Users\ARYAN SATPATHY\Downloads\pixil-frame-0 (1).png'
img = cv2.imread(path)

outimg = img.copy()


OBSTACLECOLOR = np.array([0, 0, 0])
STARTCOLOR = np.array((255, 0, 0), dtype = np.uint8)
ENDCOLOR = np.array((0, 255, 0), dtype = np.uint8)

'''
OBSTACLECOLOR = np.array([255, 255, 255])
STARTCOLOR = np.array((113, 204, 45), dtype = np.uint8)
ENDCOLOR = np.array((60, 76, 231), dtype = np.uint8)
'''

cv2.imshow('Map', resize(img))
cv2.waitKey(0)

cv2.destroyAllWindows()

SHAPE = img.shape[ : 2 :]

for x in range(100) :
    for y in range(100) :
        if False not in (img[(x,  y)] == STARTCOLOR) :
            Start = (x, y)
        if False not in (img[(x,  y)] == ENDCOLOR) :
            Destination = (x, y)

print(Start, Destination)

def isValid (pos) :
    try :
        return False in (img[pos] == OBSTACLECOLOR)
    except :
        print(pos)
        print(1 / 0)

def getY(x, pos1, pos2) : 
    y = pos1[1] + (x - pos1[0]) * (pos2[1] - pos1[1]) / (pos2[0] - pos1[0])
    return int(y)
def getX(y, pos1, pos2) : 
    x = pos1[0] + (y - pos1[1]) * (pos2[0] - pos1[0]) / (pos2[1] - pos1[1])
    return int(x)

def getPointList(pos1, pos2) : 
    pts = []
    if pos1[0] == pos2[0] : 
        if pos1[1] > pos2[1] :
            pos3 = pos1
            pos1 = pos2
            pos2 = pos3
        ys = range(pos1[1], pos2[1])
        pts = [(pos1[0], y) for y in ys]
    elif pos1[1] == pos2[1] : 
        if pos1[0] > pos2[0] :
            pos3 = pos1
            pos1 = pos2
            pos2 = pos3
        xs = range(pos1[0], pos2[0])
        pts = [(x, pos1[1]) for x in xs]
    else :
        if pos1[0] > pos2[0] :
            pos3 = pos1
            pos1 = pos2
            pos2 = pos3
        xs = range(pos1[0], pos2[0])
        pts = [(x, getY(x, pos1, pos2)) for x in xs]
        if pos1[1] > pos2[1] :
            pos3 = pos1
            pos1 = pos2
            pos2 = pos3
        ys = range(pos1[1], pos2[1])
        pts += [(getX(y, pos1, pos2), y) for y in ys]
    return pts

def canMakeStraightLine(pos1, pos2) : 
    pts = getPointList(pos1, pos2)
    for pt in pts : 
        if not isValid(pt) : 
            return False
    return True

class Node : 
    _id = 0
    GoalNodes = []
    def __init__(self, pos) :
        self.cameFrom = None
        self.pos = pos
        self.id = Node._id
        Node._id += 1
        self.d = math.inf
        self.connections = []
        self.addedToGraph = False
    def connect(self, node) :
        if node.id not in self.connections : 
            self.connections.append(node.id)
        return node.id
    def distance(self, pos) : 
        return math.sqrt((self.pos[0] - pos[0]) ** 2 + (self.pos[1] - pos[1]) ** 2)
 
class Graph : 
    def __init__(self, step) : 
        self.nodes = []
        self.occupied = []
        self.step = step
        self.edges = []
    def registerNode(self, node) :
        if node.pos in self.occupied : 
            return 'Position Occupied'
        if not node.addedToGraph : 
            self.nodes.append(node)
            self.occupied.append(node.pos)
            node.addedToGraph = True
        return node.id
    def findpath(self, start, goal) : 
        dist = 0
        return dist
    def makeNodeBetween(self, node1, node2) : 
        if node2.pos[0] - node1.pos[0] == 0 :
            x, y = node1.pos[0], node1.pos[1] + self.step * [-1, 1][node1.pos[1] < node2.pos[1]]
        dist = node1.distance(node2.pos)
        n = dist - self.step
        m = self.step

        if n > 0 : 
            m, n = n, m

            if n + m == 0 :
                return

            x, y = (m * node1.pos[0] + n * node2.pos[0]) / (m + n), (m * node1.pos[1] + n * node2.pos[1]) / (m + n)

            if not (0 <= x < SHAPE[0] and 0 <= y < SHAPE[1]) : return

            node = Node((int(x), int(y)))
            dist = self.step
        else :
            node = node2
        _id = self.registerNode(node)
        if False not in (img[node.pos] == ENDCOLOR) :
            Node.GoalNodes.append(node)
        node.d = node1.d + dist
        node.cameFrom = node1
        if type(_id) != str : 
            minD, cameFrom = math.inf, None 
            for _node in self.nodes : 
                if _node != node : 
                    if canMakeStraightLine(node.pos, _node.pos) :
                        d = node.distance(_node.pos)
                        if minD > _node.d + d :
                            minD = _node.d + d
                            cameFrom = _node
                        node.connect(_node)
                        _node.connect(node)
                        self.edges.append((node.pos, _node.pos))
            node.cameFrom = cameFrom
            node.d = minD
        # self.nodes.remove(node2)
        # self.occupied.remove(node2.pos)
        return _id
    def expand(self) :
        while True :  
            x = rnd.randint(0, SHAPE[0] - 1)
            y = rnd.randint(0, SHAPE[1] - 1)

            try : 
                if not isValid((x, y)) : 
                    # print(x, y)
                    continue
            except :
                print((x, y))
                exit()
            # closestNode = min(self.nodes, key = lambda node : (node.d + node.distance((x, y))))
            closestNode = min(self.nodes, key = lambda node : (node.distance((x, y))))
            try :
                if not canMakeStraightLine((x, y), closestNode.pos) : 
                    # print(x, y)
                    continue
            except :
                print(getPointList((x, y), closestNode.pos))
                print("Values are  : ", (x, y), closestNode.pos)
                print(1 / 0)
            else :
                break
        node = Node((x, y))
        return self.makeNodeBetween(closestNode, node)
    def bias(self, goal) : 
        x, y = goal
        if not isValid((x, y)) : 
            return None
        # closestNode = min(self.nodes, key = lambda node : (node.d + node.distance((x, y))))
        closestNode = min(self.nodes, key = lambda node : (node.distance((x, y))))
        if not canMakeStraightLine((x, y), closestNode.pos) : 
            return
        node = Node((x, y))
        return self.makeNodeBetween(closestNode, node)

outimg = resize(outimg, 9)

def convert(pos) :
    return (pos[1] * 9, pos[0] * 9)

start = time.time()
RRTGraph = Graph(4)
startNode = Node(Start)
startNode.d = 0
RRTGraph.registerNode(startNode)

choices = [1] * 100 + [2] * 200
def iterate(n = 250) : 
    for iteration in range(n) : 
        choice = rnd.choice(choices)
        if choice == 1 : 
            RRTGraph.bias(Destination)
        else : 
            RRTGraph.expand()

iterate(250)
while len(Node.GoalNodes) == 0 : 
    iterate(20)

goalNode = Node.GoalNodes[0]

print(goalNode.d, goalNode.distance(Start))

path = []
while goalNode.cameFrom != None :
    path.append(goalNode.pos)
    goalNode = goalNode.cameFrom

path.append(goalNode.pos)

end = time.time() - start

# RRTGraph.expand()


for edge in RRTGraph.edges : 
    pos1, pos2 = edge
    cv2.line(outimg, convert(pos1), convert(pos2), (255, 100, 100), 1)
    
for node in RRTGraph.nodes : 
    pos = node.pos
    cv2.circle(outimg, convert(pos), 1, (0, 0, 255), 1)


for i in range(len(path) - 1) :
    cv2.line(outimg, convert(path[i]), convert(path[i + 1]), (51, 255, 255), 3)

print('Time taken : {}s'.format(end))

cv2.imshow('Output', outimg)
cv2.waitKey(0)

cv2.destroyWindow('Output')
