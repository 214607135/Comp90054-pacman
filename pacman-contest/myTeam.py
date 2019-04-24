# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint
import layout

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    
    CaptureAgent.registerInitialState(self, gameState)
    self.isChased    =   False

  def chooseAction(self,gameState):
    ## if the pacman is eaten ##
    if gameState.getAgentPosition(self.index) == self.start:
      self.isChased   =   False
    ## if the pacman is being chased ##
    if self.isChased:
      ## find the best path to go to the start point ##
      goalPoint =   self.start
    else:
      ## find the best dot in map ##
      goalPoint =   None

    action    =   self.astar(gameState,goalPoint)
    successor =   gameState.generateSuccessor(self.index,action)
    ## check whether pacman is chased ##
    self.checkChase(successor) 
    if len(self.getFood(gameState).asList()) <= 2:
      self.isChased   =   True
    return action

  def astar(self,gameState,goalPoint):
    
    ############### initialization ##################
    explored      =   []
    exploring     =   util.PriorityQueue()
    legalAction   =   []
    done          =   False
    exploring.push([gameState,[]],0)
    food_list     =   self.getFood(gameState).asList()
    #################################################
    while not done:
      popItem         =   exploring.pop()
      currentState    =   popItem[0]
      beforeAction    =   popItem[1]
      currentPos      =   currentState.getAgentPosition(self.index)

      ## when find the needed point ##
      if currentPos ==  goalPoint or (currentPos in food_list and not self.isChased):
        done    =   True
        return beforeAction[0]
      
      ## avoid duplicate exploration ##
      if currentPos in explored:
        continue
      else:
        explored.append(currentPos)
        legalAction   =   currentState.getLegalActions(self.index)
        
      for action in legalAction:
        successor     =   currentState.generateSuccessor(self.index,action)
        successorPos  =   successor.getAgentPosition(self.index)
        # ghost Pos
        enemies   =   [successor.getAgentState(i) for i in self.getOpponents(successor)]
        ghosts    =   [a for a in enemies if not a.isPacman and a.getPosition() != None]
        if len(ghosts) > 0:
          dists   =   [self.getMazeDistance(successorPos,a.getPosition()) for a in ghosts]
          # the distance to ghost is more important, since pacman should not get too close to ghost.
          ## if goalPoint == None: distance = distance to dots ##
          ## if goalPoint != None: distance = distance to start ##
          if min(dists) < 5:
            fx    =   self.getDistance(currentState,successorPos,goalPoint) + (-100) * min(dists)
          else:
            fx    =   self.getDistance(currentState,successorPos,goalPoint) + (-100) * 10  
        else:
          fx      =   self.getDistance(currentState,successorPos,goalPoint) + (-100) * 10
          # self.ischased   =   False

        ## the item pushed into priorityQueue is the successor + past-move ##
        item      =   [successor,beforeAction + [action]]
        exploring.push(item,fx)

  def getDistance(self,gameState,successorPos,goalPoint):
    ## if goalPoint ==  None, find the best dot;
    ## else find the path to goalPoint.
    if goalPoint  ==  None:
      food_list   =   self.getFood(gameState).asList()
      if successorPos in food_list:
        ## reward is -100 ##
        distance  =   0 + (-100)
      else:
        distance  =   min([self.getMazeDistance(successorPos,food) for food in food_list])
    else:
      ## f(n) = distance to start + distance to ghost ##
      distance    =   self.getMazeDistance(successorPos,self.start)
      ## if the distance between pacman and ghost is only 1,which means 
      ## the pacman can go back to start in 1 step if it 'punches' ghost.
      if self.getMazeDistance(gameState.getAgentPosition(self.index),self.start) > 1:
        if successorPos == self.start:
          ## fx should be the biggest value which avoid pacman go into the 
          ## place where ghost is. since if pacman go where ghost is, 
          ## the successor state becomes self.start, whose distance to 
          ## ghost is quite big.
          distance      =   99999
    return distance

  def checkChase(self,successor):
    pos       =   successor.getAgentPosition(self.index)
    enemies   =   [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts    =   [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      distance        =   min([self.getMazeDistance(pos,ghost.getPosition()) for ghost in ghosts ])
      if distance <= 5 and successor.getAgentState(self.index).isPacman:
        self.isChased   =   True
    else:
      self.isChased   =   False

class DefensiveReflexAgent(CaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    
    CaptureAgent.registerInitialState(self, gameState)
    
    self.foodMissingPos     =   None
    self.findFoodMissing    =   False

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    ## if food missed and not find invaders, go to that point ##
    findInvader       =   self.checkOpponentPacman(gameState)
    self.checkFoodMissing(gameState)
    if self.findFoodMissing and not findInvader:
      action          =   self.findPath(gameState,self.foodMissingPos)
      return action

    ## if nothing happens, run professor's code ##
    actions = gameState.getLegalActions(self.index)
    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    # print successor
    pos = successor.getAgentState(self.index).getPosition()
    ## print pos
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    
    features = self.getFeatures(gameState, action)
    
    # print "features :",features
    weights = self.getWeights(gameState, action)
    # print "weights :",weights
    
    # print features * weights
    # print "over"
    return features * weights

  def getFeatures(self, gameState, action):
    # print self.index        2 and 3 are used for defense. 
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    # print "myState:",myState
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman:
      features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    # print a.getPosition()     only 5 grids close can be seen.
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: 
      features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: 
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def checkFoodMissing(self,gameState):
    previState        =   self.getPreviousObservation()
    if previState ==  None:
      return
    previFood_list    =   self.getFoodYouAreDefending(previState).asList()
    curreFood_list    =   self.getFoodYouAreDefending(gameState).asList()
    for food in previFood_list:
      if food not in curreFood_list:
        self.foodMissingPos     =   food   
        self.findFoodMissing    =   True
    if gameState.getAgentPosition(self.index) == self.foodMissingPos:
      self.findFoodMissing    =   False
    return

  def checkOpponentPacman(self,gameState):
    actions   =   gameState.getLegalActions(self.index)
    for action in actions:
      successor   =   gameState.generateSuccessor(self.index,action)
      enemies     =   [successor.getAgentState(i) for i in self.getOpponents(successor)]
      pacmans     =   [a for a in enemies if a.isPacman and a.getPosition() != None]
      if pacmans: ## if find the pacman
        return True
    return False

  def findPath(self,gameState,goalPoint):

    ############### initialization ##################
    explored      =   []
    exploring     =   util.PriorityQueue()
    legalAction   =   []
    done          =   False
    exploring.push([gameState,[]],0)
    #################################################
    while not done:
      popItem         =   exploring.pop()
      currentState    =   popItem[0]
      beforeAction    =   popItem[1]
      currentPos      =   currentState.getAgentPosition(self.index)

      ## when find the needed point ##
      if currentPos ==  goalPoint:
        done    =   True
        return beforeAction[0]
      ## avoid duplicate exploration ##
      if currentPos in explored:
        continue
      else:
        explored.append(currentPos)
        legalAction   =   currentState.getLegalActions(self.index)
      for action in legalAction:
        successor     =   currentState.generateSuccessor(self.index,action)
        successorPos  =   successor.getAgentPosition(self.index)
        fx            =   self.getMazeDistance(successorPos,goalPoint)
        ## the item pushed into priorityQueue is the successor + past-move ##
        item          =   [successor,beforeAction + [action]]
        exploring.push(item,fx)