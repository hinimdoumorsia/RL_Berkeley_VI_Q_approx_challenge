# Approche Q-Learning avec Features Personnalisées

## Introduction

Le **Q-Learning** classique apprend une table Q pour chaque paire **(état, action)**.  
Lorsque l’espace d’états est trop grand, cette approche devient impossible à stocker et à généraliser.  

L’**Approximate Q-Learning** (ou Q-learning avec features) permet de représenter la valeur Q d’une action dans un état comme une combinaison linéaire de **features** :

$$
Q(s, a) = \sum_i w_i \cdot f_i(s, a)
$$

- \(s\) : état courant  
- \(a\) : action possible dans l’état \(s\)  
- \(f_i(s, a)\) : valeur de la feature \(i\) pour la paire \((s,a)\)  
- \(w_i\) : poids associé à la feature \(i\)  

---

## Mise à jour des poids

Lorsqu’un agent observe une transition \((s, a, r, s')\), il met à jour les poids pour réduire l’erreur entre la valeur Q estimée et l’échantillon observé :

$$
w_i \leftarrow w_i + \alpha \left[ \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{target}} - Q(s,a) \right] f_i(s,a)
$$

- \(\alpha\) : taux d’apprentissage (learning rate)  
- \(\gamma\) : facteur d’actualisation (discount factor)  
- \(r + \gamma \max_{a'} Q(s',a')\) : estimation de la valeur cible  
- \(Q(s,a)\) : estimation actuelle  

Cette mise à jour rapproche \(Q(s,a)\) de la valeur réelle observée tout en adaptant les poids des features.

---

## Étapes pour implémenter un Q-Learning Approché

1. **Définir vos features** :  
   Identifier des caractéristiques importantes pour votre environnement, par exemple :  
   - distance à l’objectif  
   - présence d’obstacles  
   - direction optimale pour atteindre la sortie  

2. **Initialiser les poids** \(w_i = 0\) au départ.  

3. **Calculer Q(s,a)** :  
   Combiner linéairement les features et leurs poids :  
   $$
   Q(s, a) = \sum_i w_i \cdot f_i(s, a)
   $$

4. **Choisir l’action** :  
   - Avec probabilité \(\epsilon\) : action aléatoire (exploration)  
   - Sinon : action qui maximise \(Q(s,a)\) (exploitation)  

5. **Mettre à jour les poids** après chaque transition avec la formule de mise à jour ci-dessus.

6. **Répéter** sur plusieurs épisodes pour converger vers une politique optimale.

---

## Fichier de stockage du code

- **`qlearningAgents.py`** : contient les classes suivantes :  
  - `QLearningAgent` : Q-learning classique  
  - `ApproximateQAgent` : Q-learning avec features personnalisées  
- Vos features personnalisées sont généralement définies dans un **extracteur de features** qui peut être appelé depuis `ApproximateQAgent`.

---

## Tester et visualiser

1. Lancer l’agent approximatif avec Pacman ou Gridworld :

```bash
python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 1000
