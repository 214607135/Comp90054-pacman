# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DefensiveAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.q_table            =   {}
    self.candidateOfAction  =   util.Queue()    # points whose need-to-be-generated successors
    self.ischased           =   False
    self.wasEaten           =   False

  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    '''
    You should change this in your own agent.
    '''
    #``````````````````````````````````````````````#
    # Q Learning method
    
    ############### initialization ##################
    time            = 0         # exploring time for every step
    alpha           = 0.5       # learning rate.
    gamma           = 0.6       # discount factor, how much important the future reward is
    actionList      = ['North','South','West','East'] 
    #################################################

    # generate the Q_table
    exploring       =   util.Queue()    # points that need to be explored
    explored        =   []              # points that have been explored
    currentState    =   self.getCurrentObservation()
    reward          =   0
    done            =   False
    exploring.push(currentState)

    #################################################

    # if food_left <=2 just go back to self.start
    if len(self.getFood(gameState).asList()) <= 2:   # means win. need go back to self zone.
      self.ischased   =   True
      # bestDist = 9999
      # print "win"
      # for action in actions:
      #   successor = self.getSuccessor(gameState, action)
      #   pos2 = successor.getAgentPosition(self.index)
      #   dist = self.getMazeDistance(self.start,pos2)
      #   if dist < bestDist:
      #     bestAction = action
      #     bestDist = dist
      # return bestAction
    #################################################

    # for every step, use BFS to explore explore 30 points.
    while not done and not exploring.isEmpty():
      currentState    = exploring.pop()
      time            = time + 1    # every time pop a successor, means one more point is explored.
      currentPoint    = currentState.getAgentState(self.index).getPosition()
      # print currentPoint
      
      explored.append(currentPoint)
      # print explored
      actions         = currentState.getLegalActions(self.index)

      if Directions.STOP in actions:
        actions.remove(Directions.STOP)

      for action in actions:
        self.candidateOfAction.push(action)
      
      if time > 30:
        done = True
      
      #################################################
      if currentPoint not in self.q_table.keys():
        self.q_table[currentPoint]  = [0,0,0,0]   # [North,South,West,East] initialization
      while not self.candidateOfAction.isEmpty():
        action            = self.candidateOfAction.pop()
        indexOfAction     = actionList.index(action)
      
        #################################################   
        # the successor after moving.
        successor         = currentState.generateSuccessor(self.index,action)
        nextActions       = successor.getLegalActions(self.index)
          
        food_List1        = self.getFood(currentState).asList()
        numberOfFood1     = len(food_List1)
        food_List2        = self.getFood(successor).asList()
        numberOfFood2     = len(food_List2)

        # find the maximal value of the successor.
        valueOfQ          = [self.evaluate(successor,nextAct) for nextAct in nextActions]
        argmaxQ           = max(valueOfQ)

        if numberOfFood1 - numberOfFood2 == 1 and not self.ischased:
          reward      = 100
        else:
          reward      = 0

        # the equation of Q-learning method
        old_value       = self.q_table[currentPoint][indexOfAction]
        new_value       = (1-alpha) * old_value + alpha * (reward + gamma*argmaxQ)
        self.q_table[currentPoint][indexOfAction]     = new_value
          
        # push the succesor into queue then explore it later
        if successor.getAgentPosition(self.index) not in explored:
          exploring.push(successor)
    
    #################################################
    # part of Q_table is generated for this step

    #################################################
    # value of epsilson is small so the chance of exploring is small.
    # pacman always choose the biggest value of direction.
    epsilson        = 0.001          # the chance to explore new direction.
    currentPoint    = gameState.getAgentState(self.index).getPosition()
    actions         = gameState.getLegalActions(self.index)
    if random.uniform(0,1) < epsilson:          # explore
      #####todo:      discover unexplored direction. remove explored direction
      bestAction          = random.choice(actions)
      indexOfAction       = actionList.index(action)
    else:                                       # exploit
      # Exploit: find the direction corresponding to the max value
      # but not the initial value which is 0.
      maxValue            = -9999
      for z in range(len(self.q_table[currentPoint])):
        if self.q_table[currentPoint][z] > maxValue and self.q_table[currentPoint][z] != 0:
          maxValue              = self.q_table[currentPoint][z]
      indexOfAction         = self.q_table[currentPoint].index(maxValue)
      bestAction            = actionList[indexOfAction]

    successor           =   gameState.generateSuccessor(self.index,bestAction)
    food_List1          =   self.getFood(gameState).asList()
    food_List2          =   self.getFood(successor).asList()
    successorPos        =   successor.getAgentPosition(self.index)
    
    #################################################
    
    # used for checking whether pacman is being chased.
    myPos = successor.getAgentState(self.index).getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts  = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos,a.getPosition()) for a in ghosts]
      # if self.index == 0:
      #   print myPos
      if not self.ischased:
        # if the pacman is being chased by ghost and the distance is smaller than 5
        if min(dists) <= 5 and successor.getAgentState(self.index).isPacman:
          # if self.index == 0:
          #   print "chased"
          self.ischased   =   True
        elif min(dists) > 5:
          self.ischased   =   False
      # elif self.ischased:
      #   print "33333333"

      # go back to the self zone
    elif len(ghosts) == 0:
      self.ischased       =   False
    
    #################################################
    
    # if pacman was eaten by ghost
    if myPos == self.start:
      self.wasEaten   =   True
    
    if self.wasEaten:
      self.ischased   =   False
      self.wasEaten   =   False
    #################################################

    # if pacman eats a dot. create a new Q_table.
    if successorPos in food_List1 and successorPos not in food_List2:
      self.q_table          =     {}

    if self.ischased:
      self.q_table          =     {}
    return bestAction

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
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()

    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    myState = successor.getAgentState(self.index)
    myPos   = myState.getPosition()
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts  = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos,a.getPosition()) for a in ghosts]
      features['distanceToOpponent']  =   min(dists)  
    else:
      features['distanceToOpponent']  =   100
    features['distanceToStart']       =   self.getMazeDistance(myPos,self.start)
    return features

  def getWeights(self, gameState, action):
    if self.ischased:
      # if self.index ==  0:
      #   print "being chased"
      return {'distanceToFood': 0, 'distanceToOpponent': 10, 'distanceToStart': -1}
    else:
      # print "start hunting"
      return {'distanceToFood': -1, 'distanceToOpponent': 0, 'distanceToStart': 0}

class DefensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start             =   gameState.getAgentPosition(self.index)
    self.findEnemy         =   False
    self.enemyPos          =   None
    self.noGhost           =   False
    self.foodmissing       =   False
    self.foodPos           =   None
    self.lengthToMisFood   =   0
    #################################################

  def chooseAction(self,gameState):
    actions   =   gameState.getLegalActions(self.index)

    # do not stop moving
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
    
    # check whether food is missing
    self.foodgone(gameState,self.lengthToMisFood)
    
    # if find the enemy, use A* search to that point. if not find enemy, check
    # whether food is decreasing. else,just random walk but not go back to 
    # previous position unless only one way to go.
    if self.findEnemy:
      path    =   self.astartSearch(gameState,self.enemyPos)
      action  =   path[0]
      
      self.checkEnemy(gameState,action)
      if not self.findEnemy:
        self.foodmissing        =   False
        self.lengthToMisFood    =   0
      return action

    elif self.foodmissing:
      path                    =   self.astartSearch(gameState,self.foodPos)
      self.lengthToMisFood    =   len(path)
      action                  =   path[0]
      self.checkEnemy(gameState,action)
      return action

    else:    
      # find the path to the previous position.
      previousState       =   self.getPreviousObservation()
      if previousState    ==    None:
        return random.choice(actions)
      else:  
        if not self.noGhost:      # there is a ghost
          previousPos         =   previousState.getAgentPosition(self.index)
          path                =   self.astartSearch(gameState,previousPos)
          # then do not let ghost go back to the previous position
          if len(path)  ==  1:    # it must be 1 but better for not.
            if path[0] in actions and len(actions) != 1:
              actions.remove(path[0])
          action    =   random.choice(actions)
          # do not go to the opponent zone.
          self.checkNoGhost(gameState,action)
          if self.noGhost:
            actions.remove(action)
            self.noGhost    =   False
            if actions:
              action    =   random.choice(actions)
            else:
              action    =   Directions.STOP
          
          # check there is an enemy.
          self.checkEnemy(gameState,action)
          return action

  # check whether food decreases.
  def foodgone(self,gameState,DistanceToMisFood):
    preState    =   self.getPreviousObservation()
    currentFood =   self.getFoodYouAreDefending(gameState).asList()
    if preState != None:
      preFood   =   self.getFoodYouAreDefending(preState).asList()
      for food in preFood:
        if food not in currentFood:
          self.foodmissing    =   True
          self.foodPos        =   food
          return 
        else:
          if DistanceToMisFood == 1:
            self.foodmissing  =   False
            return 
          elif DistanceToMisFood > 1:
            self.foodmissing  =   True
            return 

  def checkNoGhost(self,gameState,action):
    if  self.getSuccessor(gameState,action).getAgentState(self.index).isPacman:
      self.noGhost    =   True
    else:
      self.noGhost    =   False

  # check whether there is an enemy
  def checkEnemy(self,gameState,action):
    successor    =   self.getSuccessor(gameState,action)
    myPos       =   successor.getAgentPosition(self.index)
    enemies     =   [successor.getAgentState(i) for i in self.getOpponents(successor)]
    pacmans     =   [a for a in enemies if a.isPacman and a.getPosition() != None]
    if pacmans:   # not empty
      self.findEnemy          =   True
      dists                   =   [self.getMazeDistance(myPos,a.getPosition()) for a in pacmans]
      minDistance             =   min(dists)
      index                   =   dists.index(minDistance)
      self.enemyPos           =   pacmans[index].getPosition()
    else:
      self.findEnemy          =   False

  def astartSearch(self,gameState,goalPoint):

    #``````````````````````````````````````````````#
    # A star search: used to find the path from gameState Pos
    # to goalPoint.
    
    ############### initialization ##################
    myPos         =   gameState.getAgentPosition(self.index)
    explored      =   []
    exploring     =   util.PriorityQueue()
    legalAction   =   []
    exploring.push([gameState,[]],0)
    done          =   False
    #################################################
      
    # A star search: f(x) = g(x) + h(x), where g(x) is the distance to 
    # successor, h(x) is the distance from successor to goalPos.
    while not done:
      popItem         =   exploring.pop()
      currentState    =   popItem[0]
      beforeAction    =   popItem[1]
      
      # when find the point we need.
      if currentState.getAgentPosition(self.index) == goalPoint:
        done              =   True
        bestAction_list   =   beforeAction
        break

      # avoid duplicate exploration.
      if currentState in explored:
        continue
      else:
        explored.append(currentState)
        legalAction   =   currentState.getLegalActions(self.index)
      for action in legalAction:
        successor     =   currentState.generateSuccessor(self.index,action)
        successorPos  =   successor.getAgentPosition(self.index)
        # the item pushed into priorityQueue is the successor + past-move.
        item          =   [successor,beforeAction + [action]]
        # fx is the priority of the exploring. fx smaller, will be explored earlier.
        fx            =   self.getMazeDistance(myPos,successorPos) + self.getMazeDistance(successorPos,goalPoint)
        exploring.push(item,fx)

    return bestAction_list

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
