# qlearningAgents.py
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import SimpleExtractor
import random, util

class QLearningAgent(ReinforcementAgent):
    """
    Q-Learning classique
    """
    def __init__(self, **args):
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()  # Q-values initialisées à 0

    def getQValue(self, state, action):
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max([self.getQValue(state, a) for a in actions])

    def computeActionFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return None
        max_value = self.computeValueFromQValues(state)
        best_actions = [a for a in actions if self.getQValue(state, a) == max_value]
        return random.choice(best_actions)

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
        Mise à jour classique Q-Learning:
        Q(s,a) ← (1-α) Q(s,a) + α [r + γ max_a' Q(s',a')]
        """
        sample = reward + self.discount * self.computeValueFromQValues(nextState)
        self.qValues[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    """
    QLearningAgent avec paramètres par défaut pour Pacman
    """
    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
    Q-Learning Approché utilisant un extracteur de features
    Q(s,a) = Σ_i w_i * f_i(s,a)
    """
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getQValue(self, state, action):
        """
        Q(s,a) = Σ_i w_i * f_i(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        return sum(self.weights[f] * features[f] for f in features)

    def update(self, state, action, nextState, reward):
        """
        Mise à jour des poids:
        w_i ← w_i + α [ (r + γ max_a' Q(s',a')) - Q(s,a) ] * f_i(s,a)
        """
        features = self.featExtractor.getFeatures(state, action)
        correction = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        for f in features:
            self.weights[f] += self.alpha * correction * features[f]

    def final(self, state):
        """Appelé à la fin de chaque partie"""
        PacmanQAgent.final(self, state)
        if self.episodesSoFar == self.numTraining:
            print("Poids finaux:", self.weights)
