# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import math
from queue import PriorityQueue

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance


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
    is another abstract class.
    """

    def __init__(self, evalFn = 'betterEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alphabeta(state):
            bestValue, bestAction = None, None
            # print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            # print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "* YOUR CODE HERE *"
        # util.raiseNotDefined()
        def alphabeta(state, alpha=None, beta=None):
            bestValue, bestAction = None, None
            for action in state.getLegalActions(0):
                succ = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            return bestAction

        def minValue(state, agentIdx, depth, alpha, beta):
            if agentIdx == state.getNumAgents():  
                return maxValue(state, 0, depth + 1, alpha, beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
                if (alpha is not None and value <= alpha):
                    return value
                if beta is None:
                    beta = value
                else:
                    beta = min(beta, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        def maxValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                if (beta is not None and value >= beta):
                    return value
                if alpha is None:
                    alpha = value
                else:
                    alpha = max(alpha, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = alphabeta(gameState)
        return action
        

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        def expValue(state, agentIdx=0, depth=0):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            succs = []
            for action in state.getLegalActions(agentIdx):
                succs.append(state.generateSuccessor(agentIdx, action))
            temp = 0
            for succ in succs:
                if agentIdx > 1:
                    temp += expValue(succ, agentIdx-1, depth)
                else:
                    temp += maxValue(succ, depth-1)
            return float(temp) / len(succs)
        
        def maxValue(state, depth=0):
            if state.isLose() or state.isWin() or depth == 0:
                return self.evaluationFunction(state)
            val = float("-inf")
            for action in state.getLegalActions():
                succ = state.generateSuccessor(0, action)
                val = max(val, expValue(succ, state.getNumAgents()-1, depth))
            return val

        legalActions = gameState.getLegalActions()
        bestAction = Directions.STOP
        val = float("-inf")
        for action in legalActions:
            succ = gameState.generateSuccessor(0, action)
            tmp = expValue(succ, gameState.getNumAgents()-1, self.depth)
            if tmp > val:
                val = tmp
                bestAction = action
        return bestAction
        
def dijkstra(pacmanPos, walls):
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    map = walls.copy()
    for i in range(walls.width):
        for j in range(walls.height):
            map[i][j] = 1e5
    map[pacmanPos[0]][pacmanPos[1]] = 0
    q = PriorityQueue()
    q.put([map[pacmanPos[0]][pacmanPos[1]], pacmanPos])
    while not q.empty():
        temp = q.get()
        d_v = temp[0]
        v = temp[1]
        if map[v[0]][v[1]] != d_v:
            continue
        for i in range(4):
            x = v[0] + dx[i]
            y = v[1] + dy[i]
            if (x >= 0 and x < walls.width and y >= 0 and y < walls.height 
                and walls[x][y] == False):
                if(map[x][y] > d_v + 1):
                    map[x][y] = d_v + 1
                    q.put([map[x][y], [x, y]])
    return map

def ghostScore(ghostStates, map):
    score = 0
    dangerGhost = 0
    edibleGhost = 0
    for ghost in ghostStates:
        v = ghost.getPosition()
        d = map[int(v[0])][int(v[1])]
        t = ghost.scaredTimer
        if (t == 0) and (d <= 3):
            score += (max(3 - d, 0))**2
            dangerGhost += 1
        elif t > d: 
            score -= (t - d)**3
            edibleGhost += 1 
    if dangerGhost == 0: 
        score = 2000
    return score - 100*dangerGhost - 50*edibleGhost 

def foodScore(food_grid, map):
    score = 0
    foods = food_grid.asList()
    if foods:
        score -= min([map[int(food[0])][int(food[1])] for food in foods])
    return score - 20*len(foods)

def capsuleScore(capsules, map):
    score = 0
    if capsules:
        score -= sum([map[int(caps[0])][int(caps[1])] for caps in capsules])
    return score - 100*len(capsules)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacman_pos = currentGameState.getPacmanPosition()
    ghost_states = currentGameState.getGhostStates()
    food_grid = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls()
    
    map = dijkstra(pacman_pos, walls)
    
    return (
        10 * currentGameState.getScore()
        + ghostScore(ghost_states, map)
        + foodScore(food_grid, map)
        + 10 * capsuleScore(capsules, map)
    )
    
# Abbreviation
better = betterEvaluationFunction
