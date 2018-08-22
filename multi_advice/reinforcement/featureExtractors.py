# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        #print "features",features;
        return features


    def getAdviceCond(self, state,action): #new def added by me to check when to take advice
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        #features = util.Counter()

        #features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        numghost=0
        dx, dy = Actions.directionToVector(action)
        steps=int(4)
        next_x, next_y = int(x + steps*dx), int(y + steps*dy)

            # count the number of ghosts 1-step away
        numghost += sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        return numghost
        # if there is no danger of ghosts then add the food feature
        #if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            #features["eats-food"] = 1.0

        #dist = closestFood((next_x, next_y), food, walls)
        #if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            #features["closest-food"] = float(dist) / (walls.width * walls.height)
        #features.divideAll(10.0)
        #print "features",features;
        #return features  


    def allFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features["all-food"] = food
        features["walls"] = walls
        features["ghosts"] = ghosts
        features.divideAll(10.0)
        #print "features",features;
        return features


    def stateFeatures(self,state):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0
        #state.pos gives the current position
        #state.direction gives the travel vector
        pacmanstate = state.getPacmanState()
        features["position"] = state.getPacmanPosition()
        features["direction"] = pacmanstate.getDirection()
        #x,y = state.getPacmanPosition()
        #dx, dy = pacmanstate.getDirection()
        #next_x, next_y = int(x + dx), int(y + dy)
        features["all-food"] = food
        features["walls"] = walls
        features["ghosts"] = ghosts
        features["vector-dir"] = pacmanstate.directionToVector(pacmanstate.getDirection())
        #features.divideAll(10.0)
        #print "state features for advice",features
        return features



class AllExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    + all the other features
    """

    def allFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

        # if there is no danger of ghosts then add the food feature
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features["all-food"] = food
        features["walls"] = walls
        features["ghosts"] = ghosts
        features.divideAll(10.0)
        #print "features",features;
        return features


    def stateFeatures(self,state):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0
        #state.pos gives the current position
        #state.direction gives the travel vector
        pacmanstate = state.getPacmanState()
        features["position"] = pacmanstate.pos
        features["direction"] = pacmanstaten.direction
        x,y = state.getPacmanPosition()
        dx, dy = pacmanstate.direction
        next_x, next_y = int(x + dx), int(y + dy)
        features["all-food"] = food
        features["walls"] = walls
        features["ghosts"] = ghosts
        #features.divideAll(10.0)
        #print "state features for advice",features
        return features

    def getFeatures(self, state, action):
        # extract the grid of food and wall locations and get the ghost locations
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()

        features["bias"] = 1.0

        # compute the location of pacman after he takes the action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pos_g = state.getGhostPositions()
        #print "ghost pos",state.getGhostPositions()
        #xg =[]
        #yg =[]
        distg =[]
        for i in range(0, len(pos_g)) :
            #xg.append(pos_g[i][0])
            #yg.append(pos_g[i][1])
            distg.append(LA.norm([x-pos_g[i][0], y-pos_g[i][1]]))
        
        #print "xg and yg" , xg, yg    
        """
        for i in range(0, len(xg)):
            distg.append(LA.norm([x-xg[i], y-yg[i]]))
        """
        features["min-ghost-dist"] = min(distg)/ (walls.width * walls.height)

        # count the number of ghosts 1-step away
        ##features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        tempfeature = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        # if there is no danger of ghosts then add the food feature
        if not tempfeature and food[next_x][next_y]:
            features["eats-food"] = 1.0
        else:
            features["eats-food"] = 0
        
        #print"get capsules" ,state.getCapsules()
        #xf = []
        #yf = [] 
        #xf,yf = state.getCapsules()


        #distf= []
        #for i in range(0, len(xf)):
        #    distf.append(LA.norm([x-xf[i],y-yf[i]]))
        #features["min-food-dist"] = min(distf)


        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # make the distance a number less than one otherwise the update
            # will diverge wildly
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        features.divideAll(10.0)
        #features["all-food"] = food
        #features["walls"] = walls
        #features["ghosts"] = ghosts
        #features.divideAll(10.0)
        #print "features",features;
        return features
