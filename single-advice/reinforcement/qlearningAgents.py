# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

import numpy as np
import random, util, math
import csv


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        "*** YOUR CODE HERE ***"
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)
        if possibleActions:
            maxv = float("-inf")
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
            return maxv
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)

        if possibleActions:
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            #print "doing self-action",bestAction,"with qvalue ",maxv,"\n"  #comment this for no printing
            return bestAction
            
        return None





    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        possibleActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if possibleActions:
            if util.flipCoin(self.epsilon) == True:
                #action = random.choice(possibleActions)
                #trying to introduce this in getpolicydist() function
                action = random.choice(possibleActions)  # random ,code might be wrong this line
                #print "doing random action ",action,"\n"
            else:
                action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        possibleActions = self.getLegalActions(nextState)
        R = reward
        if possibleActions:
            Q = []
            for a in possibleActions:
                Q.append(self.getQValue(nextState, a))
            R = reward + self.discount * max(Q)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (R - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

"""
class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.15, gamma=0.8, alpha=0.2, numTraining=0, extractor = 'SimpleExtractor', **args):
        
      #  These default parameters can be changed from the pacman.py command line.
       # For example, to change the exploration rate, try:
        #    python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        #alpha    - learning rate
        #epsilon  - exploration rate
        #gamma    - discount factor
        #numTraining - number of training episodes, i.e. no learning after these many episodes
        
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        self.featExtractor = util.lookup(extractor, globals())()



    def getAction(self, state):
        
      #  Simply calls the getAction method of QLearningAgent and then
      #  informs parent of action for Pacman.  Do not change or remove this
      #  method.plt.figure(1)
        
        action = QLearningAgent.getAction(self, state)
        #action = self.getAction(state)
        self.doAction(state, action)
        return action

    #############################################################
    '''
    def getAction(self, state):
        
      #    Compute the action to take in the current state.  With
      #    probability self.epsilon, we should take a random action and
      #    take the best policy action otherwise.  Note that if there are
      #    no legal actions, which is the case at the terminal state, you
      #    should choose None as the action.

      #    HINT: You might want to use util.flipCoin(prob)
      #    HINT: To pick randomly from a list, use random.choice(list)
        
        # Pick Action
        possibleActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if possibleActions:
            if util.flipCoin(self.epsilon) == True:
                action = random.choice(possibleActions)
            else:
                action = self.getPolicy(state)
        return action

    def update(self, state, action, nextState, reward):
        
       #   The parent class calls this to observe a
       #   state = action => nextState and reward transition.
       #   You should do your Q-Value update here
       #
       #   NOTE: You should never call this function,
       #   it will be called on your behalf
        
        possibleActions = self.getLegalActions(nextState)
        R = reward
        if possibleActions:
            Q = []
            for a in possibleActions:
                Q.append(self.getQValue(nextState, a))
            R = reward + self.discount * max(Q)
        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (R - self.getQValue(state, action))


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
    '''

    def computeActionFromQValues(self, state):
        
        #  Compute the best action to take in a state.  Note that if there
        #  are no legal actions, which is the case at the terminal state,
        #  you should return None.
        
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)

        #incorporate advice
        
       # assuming advice is prob distribution of actions corresponding to a specific state then we can add these into generating the best possible action.
        
        
        qpolicy = self.getpolicydist(state)
        advicepolicy = self.getAdvice(state,'/home/starshipcrew/reinforcement/advices/user2.csv')
        finalpolicy = util.Counter()

        #for act, prob in qpolicy.items():
        #    finalpolicy[act] = float(qpolicy[act] * finalpolicy[act])
        finalpolicy = {k: qpolicy[k]*advicepolicy[k] for k in qpolicy}

        #print " final policy ",finalpolicy
        #print " advice used"
        if possibleActions:
            #the normal way
            
        #    maxv = float("-inf")
        #    bestAction = None
        #    for action in possibleActions:
        #       q = self.getQValue(state, action)
        #        if q >= maxv:
        #            maxv = q
        #            bestAction = action
            

            #the new way 
            #check if all are equal, then select random action
            tact = finalpolicy.keys()
            tkey = finalpolicy.values()
            temp = util.Counter()
            ta = []
            #ta.append(finalpolicy.keys()[0])
            tk = []
            #tk.append(tkey[0])

            '''
            if all(x == tact[0] for x in tact):
                print " all equal, select random action"
                rint = random.randint(0,len(tkey))
                print tkey[rint]
                return tkey #random action
            '''
            maxv = tkey[0]
            bestAction = None

            for act, prob in finalpolicy.items():
                if prob > maxv:
                    bestAction = act
                    maxv = prob

            for act, prob in finalpolicy.items():
                if prob == maxv:
                    ta.append(act)        
                    tk.append(prob)

            #print "all these actions have same prob, hopefully" , ta ,tk     
            if len(ta)>0:
            #    print " random act" 
                bestAction = str(ta[random.randint(0,len(ta)-1)])
                

            #print "\n bestAction" , bestAction
            return bestAction
        return None

    def getpolicydist(self, state):
        possibleActions = self.getLegalActions(state)
        #policydist = np.zeros(4);
        policydist = util.Counter() # A Counter is a dict with default 0
        for dir, vec in Actions._directionsAsList:
            policydist[dir] = 0
        #print " init zero distr",policydist
            
        boltzsum = float(0)

        if util.flipCoin(self.epsilon) == True:
            for a in possibleActions:
                policydist[a] = float(1)/len(possibleActions)
            #print " running random policy"

        else:
            for a in possibleActions:
                boltzsum = boltzsum + np.exp(self.getQValue(state,a))

            for a in possibleActions:
                policydist[a] = np.exp(self.getQValue(state,a)) / boltzsum
            #print "running qpolicy"
        ##print "qpolicy distribution", policydist  
        #print "return q policydist"
        return policydist 

    def getAdvice(self,state,userfilename):
        advice = util.Counter()

        #single row advice only for now otherwise need a loop on this
        reader = csv.reader(open(userfilename, 'r'))
        for row in reader:
            a, b, c, d = row
            advice[a] = b
            advice[c] = d
        #print advice
        allfeatures = self.featExtractor.stateFeatures(state)
        print " features for advice", allfeatures

        #this needs a more general framework
        if advice["feature"] == "Facing-ghost":
            #run function  
            policymultiplier = self.facingGhostAdvice(state, allfeatures, advice)
          ##  print "policy multiplier",policymultiplier
        #print "return advice policy"  
        return policymultiplier    

    def facingGhostAdvice(self, state, feature, advice):
        #stepsghostaway = util.Counter()

        x, y = feature["position"]
        direction = feature["direction"]
        #dx, dy = feature["directionvector"]
        pacmanstate = state.getPacmanState()
        dx, dy = pacmanstate.directionToVector(direction)
        walls = feature["walls"]
        ghosts = feature["ghosts"]
        steps = int(advice["value"])
        next_x, next_y = int(x + steps*dx), int(y + steps*dy)
        #next2_x, next2_y = int(x + 2*dx), int(y + 2*dy)
        #next3_x, next3_y = int(x + 3*dx), int(y + 3*dy)

        # count the number of ghosts n-step away
        ghostsaway = sum((next_x, next_y) in pacmanstate.getLegalNeighbors(g, walls) for g in ghosts)
        #stepsghostaway["2"] = sum((next2_x, next2_y) in state.getLegalNeighbors(g, walls) for g in ghosts)
        #stepsghostaway["3"] = sum((next3_x, next3_y) in state.getLegalNeighbors(g, walls) for g in ghosts)

        policymultiplier = util.Counter() # A Counter is a dict with default 0
        for dir, vec in Actions._directionsAsList:
            policymultiplier[dir] = 1
        
        if ghostsaway >= 1:
            policymultiplier[direction] = 0.1                   #direction variable or dir and in loop or not and which direction 

        return policymultiplier

    '''    
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
    '''
"""

class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action





class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.lweights=util.Counter()
        import csv
        with open('/home/sunny/Desktop/intern_30_07/solvedpacman/single-advice/learned_classic1.csv') as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            for row in csvReader:
                self.lweights[row[0]]=float(row[1])
            for feature in self.lweights.keys():
                print " ",feature,"  ",self.lweights[feature]
    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        qv = 0
        for feature in f:
            qv = qv + self.weights[feature] * f[feature]
        return qv

    def getlQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        f = self.featExtractor.getFeatures(state, action)
        qv = 0
        for feature in f:
            qv = qv + self.lweights[feature] * f[feature]
        return qv
      

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        R = reward
        f = self.featExtractor.getFeatures(state, action)
        alphadiff = self.alpha * ((R + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        #alphadiff = self.alpha * ((R + self.discount * self.getValue(nextState)) - self.getQValue(state, action))
        for feature in f.keys():
            self.weights[feature] = self.weights[feature] + alphadiff * f[feature]


    def getAction(self,state):

        possibleActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        numghosts=0
        for action in ["North","South","East","West"]:
            #print action
            numghosts+= self.featExtractor.getAdviceCond(state,action)
        if possibleActions:

            if numghosts and (self.epsilon <> 0.0) == True:  #action by advice at random points , give at crucial points
                maxv = float("-inf")
                bestAction = None
                for action in possibleActions:
                    q = self.getlQValue(state, action)  # use learned weights, see the differnce in functions
                    if q >= maxv:
                        maxv = q
                        bestAction = action
            #return bestAction
                #print "learned_weights-using\n"
               # print "doing advice-action ",action," with q value", maxv,"\n"
                self.doAction(state, bestAction)
            else:
                #print "self_weights-using ",self.epsilon,"\n"
                action = QLearningAgent.getAction(self,state)     #Q-learning(include boltzmann later)
                self.doAction(state, action)
                
            #action = random.choice(possibleActions)  
        else:
            #print "self_weights-using ",self.epsilon,"\n"
            action = QLearningAgent.getAction(self,state)     #Q-learning(include boltzmann later)
            self.doAction(state, action)
            

        #self.doAction(state, action)
        return action

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        import csv
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"

            for feature in self.weights.keys():
                print " ",feature,"  ",self.weights[feature]  #for seeing values of weights
            w = csv.writer(open(('/home/sunny/Desktop/intern_30_07/solvedpacman/single-advice/reinforcement/output/output' + '-'.join([str(t) for t in time.localtime()[1:6]]) + '.csv'), "w"))            #writing learned weights in file
            for key, val in self.weights.items():
                w.writerow([key, val])
            pass

           
"""           

    def computeActionFromQValues(self, state):
        
      #    Compute the best action to take in a state.  Note that if there
      #    are no legal actions, which is the case at the terminal state,
       #   you should return None.
        
        "*** YOUR CODE HERE ***"
        possibleActions = self.getLegalActions(state)

        #incorporate advice
        
      #  assuming advice is prob distribution of actions corresponding to a specific state then we can add these into generating the best possible action.
        
        
        qpolicy = self.getpolicydist(state)
        advicepolicy = self.getAdvice(state,'/home/starshipcrew/reinforcement/advices/user4.csv')  #advice value  = 1
        finalpolicy = util.Counter()

        #for act, prob in qpolicy.items():
        #    finalpolicy[act] = float(qpolicy[act] * finalpolicy[act])
        finalpolicy = {k: qpolicy[k]*advicepolicy[k] for k in qpolicy}

        ##print " final policy ",finalpolicy

        if possibleActions:
            #the normal way
            
            maxv = float("-inf")
            bestAction = None
            for action in possibleActions:
                q = self.getQValue(state, action)
                if q >= maxv:
                    maxv = q
                    bestAction = action
            

            #the new way 
            maxv = float("-inf")
            bestAction = None
            for act, prob in finalpolicy.items():
                if prob >= maxv:
                    bestAction = act
                    maxv = prob  

            print " bestAction" , bestAction," ",state
            return bestAction
        return None




       

    def getpolicydist(self, state):
        possibleActions = self.getLegalActions(state)
        #policydist = np.zeros(4);
        policydist = util.Counter() # A Counter is a dict with default 0
        for dir, vec in Actions._directionsAsList:
            policydist[dir] = 0
        #print " init zero distr",policydist
            
        boltzsum = float(0)
        for a in possibleActions:
            boltzsum = boltzsum + np.exp(self.getQValue(state,a))

        for a in possibleActions:
            policydist[a] = np.exp(self.getQValue(state,a)) / boltzsum

        ##print "qpolicy distribution", policydist  
        return policydist 

    def getAdvice(self,state,userfilename):
        advice = util.Counter()

        #single row advice only for now otherwise need a loop on this
        reader = csv.reader(open(userfilename, 'r'))
        for row in reader:
            a, b, c, d = row
            advice[a] = b
            advice[c] = d
        #print advice
        allfeatures = self.featExtractor.stateFeatures(state)
        ##print " features for advice", allfeatures

        #this needs a more general framework
        if advice["feature"] == "Facing-ghost":
            #run function  
            policymultiplier = self.facingGhostAdvice(state, allfeatures, advice)
          ##  print "policy multiplier",policymultiplier

        return policymultiplier    

    def facingGhostAdvice(self, state, feature, advice):
        #stepsghostaway = util.Counter()

        x, y = feature["position"]
        direction = feature["direction"]
        #dx, dy = feature["directionvector"]
        pacmanstate = state.getPacmanState()
        dx, dy = pacmanstate.directionToVector(direction)
        walls = feature["walls"]
        ghosts = feature["ghosts"]
        steps = int(advice["value"])
        ghostsaway = 0

        for i in range(1,steps):
            next_x, next_y = int(x + i*dx), int(y + i*dy)


            ###big big loophole FIX THIS !!!!!!!!

            #next2_x, next2_y = int(x + 2*dx), int(y + 2*dy)
            #next3_x, next3_y = int(x + 3*dx), int(y + 3*dy)

            # count the number of ghosts n-step away
            ghostsaway = ghostsaway + sum((next_x, next_y) in pacmanstate.getLegalNeighbors(g, walls) for g in ghosts)
            #stepsghostaway["2"] = sum((next2_x, next2_y) in state.getLegalNeighbors(g, walls) for g in ghosts)
            #stepsghostaway["3"] = sum((next3_x, next3_y) in state.getLegalNeighbors(g, walls) for g in ghosts)

        policymultiplier = util.Counter() # A Counter is a dict with default 0
        for dir, vec in Actions._directionsAsList:
            policymultiplier[dir] = 1
        
        if ghostsaway >= 1:
            policymultiplier[direction] = 0.1
        print policymultiplier
        return policymultiplier    

"""






class NeuralNetQAgent(PacmanQAgent):
    def __init__(self, extractor='IdentityExtractor', *args, **kwargs):
        self.nnet = None
        PacmanQAgent.__init__(self, *args, **kwargs)

    def getQValue(self, state, action):
        if self.nnet is None:
            self.nnet = NeuralNetwork(state)
        prediction = self.nnet.predict(state, action)
        return prediction

    def update(self, state, action, nextState, reward):
        if self.nnet is None:
            self.nnet = NeuralNetwork(state)

        maxQ = 0
        for a in self.getLegalActions(nextState):
            if self.getQValue(state, action) > maxQ:
                maxQ = self.getQValue(state, action)

        y = reward + (self.discount * maxQ)

        self.nnet.update(nextState, action, y)


class NeuralNetwork:
    def __init__(self, state):
        walls = state.getWalls()
        self.width = walls.width
        self.height = walls.height
        self.size = 5 * self.width * self.height

        self.model = Sequential()
        tempaction = 'East'
        reshaped_state = self.reshape(state, tempaction)
        print "input shape", reshaped_state.shape
        #self.model.add(Dense(164, init='lecun_uniform', input_shape=(875,)))
        self.model.add(Dense(164, init='lecun_uniform', input_shape=(reshaped_state.shape[1],)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(150, init='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(1, init='lecun_uniform'))
        self.model.add(Activation('linear'))

        rms = RMSprop()
        self.model.compile(loss='mse', optimizer=rms)

    def predict(self, state, action):
        reshaped_state = self.reshape(state, action)
        return self.model.predict(reshaped_state, batch_size=1)[0][0]

    def update(self, state, action, y):
        reshaped_state = self.reshape(state, action)
        y = [[y]]
        self.model.fit(reshaped_state, y, batch_size=1, nb_epoch=1, verbose=1)

    def reshape(self, state, action):
        reshaped_state = np.empty((1, 2 * self.size))
        food = state.getFood()
        walls = state.getWalls()
        for x in range(self.width):
            for y in range(self.height):
                reshaped_state[0][x * self.width + y] = int(food[x][y])
                reshaped_state[0][self.size + x * self.width + y] = int(walls[x][y])
        ghosts = state.getGhostPositions()
        ghost_states = np.zeros((1, self.size))
        for g in ghosts:
            ghost_states[0][int(g[0] * self.width + g[1])] = int(1)
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        pacman_state = np.zeros((1, self.size))
        pacman_state[0][int(x * self.width + y)] = 1
        pacman_nextState = np.zeros((1, self.size))
        pacman_nextState[0][int(next_x * self.width + next_y)] = 1

        #print " initial reshape",reshaped_state
        #print "\n ghost states",ghost_states
        #print " \n pacaman_states",pacman_state
        #print "\n pacman next state",pacman_nextState
        reshaped_state = np.concatenate((reshaped_state, ghost_states, pacman_state, pacman_nextState), axis=1)
        #print "\n reshaped state combined ", reshaped_state
        return reshaped_state
