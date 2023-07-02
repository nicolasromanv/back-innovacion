import json
import os
import random

import numpy as np

from keras.layers import (
    Input,
    Embedding,
    Dot,
    Reshape,
    Dense
)
from keras.models import Model
import tensorflow as tf
from imagenjuego import obtener_caratula

random.seed(100)

def generate_batch(pairs, n_positive = 50, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (game_id, tag_id) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (game_id, tag_id, 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_game = random.randrange(len(game_index))
            random_tag = random.randrange(len(tag_index))
            
            # Check to make sure this is not a positive example
            if (random_game, random_tag) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_game, random_tag, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'game': batch[:, 0], 'tag': batch[:, 1]}, batch[:, 2]

def game_embedding_model(embedding_size = 100, classification = False):
    """Model to embed game and tags using the functional API.
       Trained to discern if a tag is present for a game"""
    
    # Both inputs are 1-dimensional
    game = Input(name = 'game', shape = [1])
    tag = Input(name = 'tag', shape = [1])
    
    # Embedding the game (shape will be (None, 1, 50))
    game_embedding = Embedding(name = 'game_embedding',
                               input_dim = len(game_index),
                               output_dim = embedding_size)(game)
    
    # Embedding the tag (shape will be (None, 1, 50))
    tag_embedding = Embedding(name = 'tag_embedding',
                               input_dim = len(tag_index),
                               output_dim = embedding_size)(tag)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([game_embedding, tag_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [book, link], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [game, tag], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model


# Method to find the closest matches to the game in question
def find_closest(game_embedding: np.array) -> None:
    dists = np.dot(game_weights, game_embedding)
    sorted_dists = np.argsort(dists)
    closest = sorted_dists[-6:]
    resultados = []
    for c in reversed(closest):
        resultado = {
            'game': index_game[c],
            'similarity': dists[c]
        }
        resultados.append(resultado)
    return resultados



def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # Normalize
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

def subtract_tag(tag: str, game: str) -> np.array:
    """
    Subtracts a tag embedding from a game embedding and normalises
    
    :type tag: str
    :param tag: Tag to subtract from game embedding
    :type game: str
    :param game: Game which tag embedding is subtracted from
    :rtype: np.array
    :return: New game array with the tag embedding removed
    """
    new_game_weight = game_weights[game_index[game]] - tag_weights[tag_index[tag]]
    return new_game_weight / np.linalg.norm(new_game_weight).reshape((-1, 1))[0]

def add_tag(tag: str, game: str) -> np.array:
    """
    Adds a tag embedding from a game embedding and normalises
    
    :type tag: str
    :param tag: Tag to add to game embedding
    :type game: str
    :param game: Game which tag embedding is added to
    :rtype: np.array
    :return: New game array with the tag embedding added
    """
    new_game_weight = game_weights[game_index[game]] + tag_weights[tag_index[tag]]
    return new_game_weight / np.linalg.norm(new_game_weight).reshape((-1, 1))[0]


#--------------------------------
os.chdir('..')
random.seed(40)
np.random.seed(40)


with open("games_with_tags_double_filter.json", 'r') as in_json:
    games_with_tags = json.load(in_json)

game_index = {game: idx for idx, game in enumerate(games_with_tags)}
index_game = {idx: game for game, idx in game_index.items()}

tag_count = 0
tag_index = {}
for game, tags in games_with_tags.items():
    for tag in tags:
        if tag not in tag_index:
            tag_index[tag] = tag_count
            tag_count += 1
index_tag = {idx: tag for tag, idx in tag_index.items()}

pairs = []
for game, tags in games_with_tags.items():
    for tag in tags:
        pairs.append(tuple((game_index[game], tag_index[tag])))

pairs_set = set(pairs)

#---------------------------

next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))
tf.random.set_seed(42)



#-------------------------------------

model = game_embedding_model(embedding_size=400)
"""SI LAS SIMILITUDES SON BAJAS, ACTIVAR EL ENTRENAMIENTO NUEVAMENTE!
n_positive = 1024

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(
    gen,
    epochs = 20, 
    steps_per_epoch = len(pairs) // n_positive,
    verbose = 2
)
model.save('double_filter_emb_200.h5')
"""

# Extract embeddings
game_layer = model.get_layer('game_embedding')
game_weights = game_layer.get_weights()[0]

game_weights = game_weights / np.linalg.norm(game_weights, axis = 1).reshape((-1, 1))
game_weights[0][:10]
np.sum(np.square(game_weights[0]))

tag_weights = extract_weights('tag_embedding', model)


juego_buscar= input("Qué juego desea buscar: ")
resultados=find_closest(game_weights[game_index[juego_buscar]])

imagenes=[]
for i in resultados:
    imagen = (obtener_caratula(i["game"]))
    imagenes.append(imagen)

resultados_finales = []
for caratula, juego in zip(imagenes, resultados):
    juego["caratula"] = caratula.get("caratula")
    resultados_finales.append(juego)

for resultado in resultados_finales:
    for clave, valor in resultado.items():
        if isinstance(valor, np.float32):
            resultado[clave] = float(valor)
with open('to-send/juegos-con-imagen.json', 'w') as archivo:
    json.dump(resultados_finales, archivo, indent=4)
