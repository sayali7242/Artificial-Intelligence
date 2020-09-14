
# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    
    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.  
  
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostState = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostState]
    foods = newFood.asList()
    score = successorGameState.getScore()
    x = newPos[0]
    y = newPos[1]
   
    """*** Your code here ***"""
    for food in foods:
    	foodDist = util.manhattanDistance(newPos, food)
    	if action != 'Stop': 
    		score += 1.0/foodDist		#Take the reciprocal of distance 
    	else:
    		score -= 1.0/foodDist
    	
    
    for ghost in newGhostState: 
    	ghostPos = ghost.getPosition()
    	ghostDist = util.manhattanDistance(newPos, ghostPos)
    	if (abs(x - ghostPos[0]) + abs(y - ghostPos[1])) > 1:
    		for scaredTime in newScaredTimes:
					if scaredTime > 2:
						score += 1.0/ghostDist
					else:
						score -= 1.0/ghostDist
    return score

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    ianother abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def minmax(self, gameState, depth, agentIndex):
  	if gameState.isWin() or gameState.isLose() or depth == 0:
  		return self.evaluationFunction(gameState)
  	elif agentIndex == 0:
  		return self.max(gameState, depth, agentIndex)
  	else: 
  		return self.min(gameState, depth, agentIndex)
  		
	def min(self, gameState, depth, agentIndex):
		minimum = []
  	legalActions = gameState.getLegalActions(agentIndex)
  	numGhosts = gameState.getNumAgents() - 1
  	for action in legalActions:
  		if agentIndex == numGhosts:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			minimum.append(self.minmax(nextGameState, depth - 1, 0))
  		else:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			minimum.append(self.minmax(nextGameState, depth, agentIndex + 1))
  	return min(minimum)
  
  def max(self, gameState, depth, agentIndex = 0):
		maximum = []
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			nextGameState = gameState.generateSuccessor(agentIndex, action)
			maximum.append(self.minmax(nextGameState, depth -1 , 1))
		return max(maximum)
  		
  def getAction(self, gameState):
  	legalActions = gameState.getLegalActions(0)
  	value = float("-inf")
  	newAction = Directions.STOP
  	for action in legalActions:
  		state = gameState.generateSuccessor(0, action)
  		newValue = self.minmax(state, 0, 1)
  		if newValue > value:
  			newAction = action
  			value = newValue
  	return newAction
  	util.raiseNotDefined()
			
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """  		
  def alphaBeta(self, gameState, depth, alpha, beta, agentIndex):
  	if gameState.isWin() or gameState.isLose() or depth == 0:
  		return self.evaluationFunction(gameState)
  	elif agentIndex == 0:
  		return self.alpha(gameState, depth, alpha, beta, agentIndex)
  	else: 
  		return self.beta(gameState, depth, alpha, beta, agentIndex)
  		
	def beta(self, gameState, depth, alpha, beta, agentIndex):
		minimum = []
  	legalActions = gameState.getLegalActions(agentIndex)
  	numGhosts = gameState.getNumAgents() - 1
  	for action in legalActions:
  		if agentIndex == numGhosts:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			minimum.append(self.minmax(nextGameState, depth - 1, alpha, beta, 0))
  		else:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			minimum.append(self.minmax(nextGameState, depth, alpha, beta, agentIndex + 1))
			if min(minimum) < alpha:
				return min(minimum)
			beta = min(beta, min(minimum))
  	return min(minimum)
  
  def alpha(self, gameState, depth, alpha, beta, agentIndex = 0):
		maximum = []
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			nextGameState = gameState.generateSuccessor(agentIndex, action) 
			maximum.append(self.minmax(nextGameState, depth - 1, alpha, beta, 1))
			if max(maximum) > beta:
				return max(maximum)
			alpha = max(alpha, max(maximum))
		return max(maximum)
  		
  		
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalActions(0)
    alpha = float("-inf")
    beta = float("inf")
    value = float("-inf")
    newAction = Directions.STOP
    for action in legalActions:
    	state = gameState.generateSuccessor(0, action)
    	newValue = self.alphaBeta(state, 0, alpha, beta, 1)
    	if newValue > value:
    		newAction = action
    		value = newValue
    	alpha = max(alpha, value)
    return newAction
    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def expectimax(self, gameState, depth, agentIndex):
  	if gameState.isWin() or gameState.isLose() or depth == 0:
  		return self.evaluationFunction(gameState)
  	elif agentIndex == 0:
  		return self.max(gameState, depth, agentIndex)
  	else: 
  		return self.avg(gameState, depth, agentIndex)
  
	def avg(self, gameState, depth, agentIndex):
		value = 0
		legalActions = gameState.getLegalActions(agentIndex)
  	numGhosts = gameState.getNumAgents() - 1
  	for action in legalActions:
  		if agentIndex == numGhosts:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			value += self.expectimax(nextGameState, depth - 1, 0)
  		else:
  			nextGameState = gameState.generateSuccessor(agentIndex, action)
  			value += self.expectimax(nextGameState, depth, agentIndex + 1)
  	return value/len(legalActions)

  def max(self, gameState, depth, agentIndex = 0):
		maximum = []
		legalActions = gameState.getLegalActions(agentIndex)
		for action in legalActions:
			nextGameState = gameState.generateSuccessor(agentIndex, action)
			maximum.append(self.minmax(nextGameState, depth - 1, 1))
		return max(maximum)
		
  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    legalActions = gameState.getLegalActions(0)
    value = float("-inf")
    newAction = Directions.STOP
    for action in legalActions:
    	state = gameState.generateSuccessor(0, action)
    	newValue = self.expectimax(state, 0, 1)
    	if newValue > value:
    		newAction = action
    		value = newValue
    return newAction
    util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"

  newPos = currentGameState.getPacmanPosition()
  newFood = currentGameState.getFood()
  foods = newFood.asList()
  newGhostState = currentGameState.getGhostStates()
  numAgents = currentGameState.getNumAgents()
  score = currentGameState.getScore()
	foodDist = float("-inf")
  for food in foods:
		maxFoodDist = max(foodDist, util.manhattanDistance(newPos, food))
		score += 1.0/maxFoodDist		#Take the reciprocal of distance 
	ghostDist = float("inf")	
	for ghost in newGhostState: 
		ghostPos = ghost.getPosition()
		minGhostPos = min(ghostDist, util.manhattanDistance(ghostPos, newPos))
		if (abs(x - ghostPos[0]) + abs(y - ghostPos[1])) > 1:
			for scaredTime in newScaredTimes:
				if scaredTime > 2:
					score += 1.0/minGhostPos
				else:
					score -= 1.0/minGhostPos
	return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

