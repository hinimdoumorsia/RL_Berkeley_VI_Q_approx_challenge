# valueIterationAgents.py
# -----------------------
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

import mdp, util
from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        A ValueIterationAgent takes a Markov decision process (MDP) on
        initialization and runs value iteration for a given number of iterations
        using the supplied discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm.
        """
        for i in range(self.iterations):
            newValues = util.Counter()  # Nouvelle table de valeurs pour l'itération k+1
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                else:
                    actions = self.mdp.getPossibleActions(state)
                    q_values = [self.computeQValueFromValues(state, action) for action in actions]
                    newValues[state] = max(q_values)
            self.values = newValues

    def getValue(self, state):
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute Q(s,a) = somme_{s'} T(s,a,s') * [R(s,a,s') + gamma * V(s')]
        """
        q_value = 0
        for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, nextState)
            q_value += prob * (reward + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          Return the best action according to current values.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        best_action = None
        max_q = float('-inf')
        for action in actions:
            q = self.computeQValueFromValues(state, action)
            if q > max_q:
                max_q = q
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
