# Projet Pac-Man – CS188  
## Question : Q-Learning Agent  

### Objectif
Apprendre une politique optimale par **essais et erreurs** dans un environnement MDP simulé. Contrairement à l’itération de valeur, l’agent **ne connaît pas le modèle complet** du MDP à l’avance.

---

### Étapes pour répondre à la question

1. **Initialisation**
   - Créer un dictionnaire `qValues` pour stocker Q(s, a), avec valeur par défaut 0.
   - Définir les hyperparamètres : `epsilon` (exploration), `alpha` (learning rate), `discount` (γ).

2. **Définir les fonctions principales**
   - `getQValue(state, action)` : retourner Q(s, a) ou 0 si jamais rencontré.
   - `computeValueFromQValues(state)` : retourner la valeur maximale Q pour l’état.
   - `computeActionFromQValues(state)` : retourner l’action qui maximise Q(s, a), avec **départage aléatoire** en cas d’égalité.
   - `update(state, action, nextState, reward)` : mettre à jour Q(s, a) selon :  
     \[
     Q(s, a) \leftarrow (1 - \alpha) Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a')]
     \]

3. **Définir l’action choisie**
   - `getAction(state)` : choisir une action ε-greedy :
     - Avec probabilité ε : action aléatoire.
     - Sinon : meilleure action selon `computeActionFromQValues`.

4. **Épisodes d’apprentissage**
   - Lancer plusieurs épisodes (`numTraining`) où l’agent interagit avec l’environnement.
   - Observer comment les valeurs Q évoluent au fil du temps.

---

### Commandes pour exécuter et tester

1. **Exécution classique du Q-learning dans Gridworld** :
```bash
python gridworld.py -a q -k 5 -m --noise 0.0
