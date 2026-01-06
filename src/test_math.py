import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1. Definiujemy filmy jako wektory ocen 3 użytkowników
# [User1, User2, User3]
shrek = np.array([[5, 5, 1]])
toy_story = np.array([[4, 5, 1]])
pila = np.array([[1, 2, 5]])

# 2. Liczymy podobieństwo (od 0 do 1)
sim_shrek_toy = cosine_similarity(shrek, toy_story)[0][0]
sim_shrek_pila = cosine_similarity(shrek, pila)[0][0]

print(f"Podobieństwo Shrek vs Toy Story: {sim_shrek_toy:.4f}")  # Powinno być blisko 1.0
print(f"Podobieństwo Shrek vs Piła:      {sim_shrek_pila:.4f}")  # Powinno być niżej
