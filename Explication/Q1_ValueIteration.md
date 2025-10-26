# Projet Pac-Man – CS188  
## Question 1 : Itération de valeur (Value Iteration)

### Équation de mise à jour

Pour un état $s$ à l’itération $k+1$ :

$$
V_{k+1}(s) = \max_{a \in Actions(s)} \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V_k(s') \right]
$$

- $V_k(s)$ : valeur de l’état $s$ à l’itération $k$  
- $T(s, a, s')$ : probabilité de transition vers l’état $s'$ en prenant l’action $a$  
- $R(s, a, s')$ : récompense reçue lors de la transition  
- $\gamma$ : facteur d’actualisation (discount factor)  
- L’itération est **batch**, donc chaque itération $k+1$ utilise les valeurs de tous les états à l’itération $k$.

---

### Valeur Q d’une action

$$
Q(s, a) = \sum_{s'} T(s, a, s') \left[ R(s, a, s') + \gamma V(s') \right]
$$

---

### Politique optimale

$$
\pi(s) = \arg\max_{a} Q(s, a)
$$

---

### Astuce pour exécuter et tester

Pour tester ton `ValueIterationAgent` :

```bash
# Lancer l'autograder pour la question 1
python autograder.py -q q1

# Visualiser Pacman avec l'agent ValueIteration
python gridworld.py -a value -i 100 -k 10
