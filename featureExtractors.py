# featureExtractors.py
from game import Directions, Actions
import util

"Feature extractors for Pacman game states"

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
        Retourne un dictionnaire de features pour Q-learning approché.
        """
        util.raiseNotDefined()


class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state, action)] = 1.0
        return feats


class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[1]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats


def closestFood(pos, food, walls):
    """
    Retourne la distance jusqu'à la nourriture la plus proche
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        if food[pos_x][pos_y]:
            return dist
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist + 1))
    return None


class SimpleExtractor(FeatureExtractor):
    """
    Feature extractor pour Approximate Q-Learning:
    - distance à la nourriture la plus proche
    - distance moyenne à la nourriture
    - nombre de nourriture proche
    - présence de fantômes proches
    - distance aux fantômes
    - arrêt ou mouvement
    - mobilité (nombre de directions légales)
    """
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()

        features = util.Counter()
        features["bias"] = 1.0

        # position après action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # fantômes à 1 pas
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )

        # distances aux fantômes
        for i, g in enumerate(ghosts):
            dist = abs(next_x - g[0]) + abs(next_y - g[1])
            features[f'dist-ghost-{i}'] = 1.0 / (dist + 1)

        # nourriture si pas de danger immédiat
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        # distance à la nourriture la plus proche
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)

        # distance moyenne à toute la nourriture
        foodList = food.asList()
        if foodList:
            avg_food_dist = sum(abs(next_x - fx) + abs(next_y - fy) for fx, fy in foodList) / len(foodList)
            features["avg-food-dist"] = avg_food_dist / (walls.width + walls.height)

            # nourriture proche (dans 2 pas)
            features["food-nearby"] = sum(abs(next_x - fx) + abs(next_y - fy) <= 2 for fx, fy in foodList)

        # arrêt ou mouvement
        features["is-stopping"] = 1.0 if action == Directions.STOP else 0.0

        # mobilité : nombre de directions légales depuis la prochaine position
        legalNext = Actions.getLegalNeighbors((next_x, next_y), walls)
        features["num-legal-next"] = len(legalNext) / 4.0  # normalisation

        # normalisation finale
        features.divideAll(10.0)
        return features
