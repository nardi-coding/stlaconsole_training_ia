from stlacore import *
from random import random, randrange, randint, shuffle
from copy import deepcopy
import pickle
import stlacore
from shutil import copyfile




base_de_donnees = "stla.sqlite"
emplacement_copie_DB = "stla.sqlite"
emplacement_permament = "stla.sqlite"

#### LES CLASSES NECESSAIRES POUR L'IA




PROBABILITIES = 0; VALUE = 1


class Noeud:
    def __init__(self, parent, action, equipe):
        self.parent = parent
        self.action = action
        self.N = 0
        self.Ns = 0
        self.quiJoue = equipe
        self.enfants = {}
        self.edges = {}
        self.index_to_action = {}

    def est_feuille(self):
        return len(self.edges) == 0

    def ajouter_enfant(self, edge_action, P, i):
        self.edges[edge_action] = {
                                    "N_s_a": 0,
                                    "W_s_a": 0,
                                    "Q_s_a": 0,
                                    "P_s_a": P,
                                    "index": i
                                }
        self.index_to_action[i] = edge_action



class NeuralNetworks:
    def __init__(self, nr_entrees, nr_d_actions, optimizer = "SDG"):
        self.reseau_probas = Network(optimizer = optimizer)
        self.reseau_values = Network(optimizer = optimizer)

        self.reseau_probas.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, 64, "sigmoid")
        self.reseau_probas.add_layer(64, nr_d_actions, "sigmoid")

        self.reseau_values.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_values.add_layer(64, 32, "sigmoid")
        self.reseau_values.add_layer(32, 1, "tanh")


    def predict(self, entree):
        vecteur_probas = self.reseau_probas.predict(entree)
        valeur_de_gain = self.reseau_values.predict(entree)
        return vecteur_probas, valeur_de_gain

"""

Created on Tuesday 20 October, 2020

@author: N.X.

"""


import numpy as np

class Optimizer:
    def __init__(self, optimizer, n_e, n_n):
        self.opt = optimizer
        if optimizer == "Adam":
            self.m_t = np.zeros((n_e, n_n))
            self.v_t = np.zeros((n_e, n_n))
            self.t = 0
            self.B1 = 0.9
            self.B2 = 0.999
            self.epsilon = 1e-8
    def optimize(self, grad):
        if self.opt == "Adam":
            self.t += 1
            self.m_t = self.B1 * self.m_t + (1 - self.B1) * grad
            self.v_t = self.B2 * self.v_t + (1 - self.B2) * grad ** 2
            m_chap = self.m_t / (1 - self.B1** self.t)
            v_chap = self.v_t / (1 - self.B2 ** self.t)
            return  m_chap/(np.sqrt(v_chap) + self.epsilon)
        elif self.opt == "SDG":
            return grad
        else:
            raise Exception("Uknown optimization method")

class Layer:
    def __init__(self, number_of_entries, number_of_neurons, activation_function, optim):
        self.weights = np.random.randn(number_of_entries, number_of_neurons) * np.sqrt(1/4)
        self.biais = np.zeros((1, number_of_neurons))
        self.activation_function = activation_function
        self.opt_p = Optimizer(optim, number_of_entries, number_of_neurons)
        self.opt_b = Optimizer(optim, 1, number_of_neurons)

    def act_fun(self, Z):
        if self.activation_function == "sigmoid" :
            return 1.0 / (1.0 + np.exp(-Z))
        elif self.activation_function == "relu":
            return np.maximum(0, Z)
        elif self.activation_function == "tanh":
            a = np.exp(Z) ** 2
            return (a - 1) / (a + 1)
        elif self.activation_function == "arctan":
            return np.arctan(Z)/np.pi + 0.5
        else:
            return Z

    def deriv_act_fun(self, Z):
        if self.activation_function == "sigmoid" :
            d = 1.0 / (1.0 + np.exp(-Z))
            return d * (1 - d)
        elif self.activation_function == "relu":
            Z[Z > 0] = 1
            Z[Z <= 0] = 0
            return Z
        elif self.activation_function == "tanh":
            a = np.exp(Z) ** 2
            return  1 - ((a - 1) / (a + 1))
        elif self.activation_function == "arctan":
            return 1/(np.pi*(1 + Z ** 2))
        else:
            return 1

    def forward(self, x):
        self.layer_before_activation = []
        self.layer_after_activation = []
        x = x.dot(self.weights) + self.biais
        self.layer_before_activation.append(x)
        x = self.act_fun(x)
        self.layer_after_activation.append(x)
        return x

    def backward(self, previous_layer, delta_l_1, eta):
        delta_l = np.dot(delta_l_1, self.weights.T)* previous_layer.deriv_act_fun(previous_layer.layer_before_activation[0])
        grad_weights = previous_layer.layer_after_activation[0].T * delta_l_1
        grad_biais = delta_l_1

        weights = self.opt_p.optimize(grad_weights)
        biais = self.opt_b.optimize(grad_biais)

        self.weights -= eta * weights
        self.biais -= eta * biais

        return delta_l

    def backward_first_layer(self, x, err, eta):
        grad_weights = x.T * err
        grad_biais = err

        weights = self.opt_p.optimize(grad_weights)
        biais = self.opt_b.optimize(grad_biais)

        self.weights -= eta * weights
        self.biais -= eta * biais

class Network:
    def __init__(self, optimizer = "SDG"):
        self.optim = optimizer
        self.layers = []

    def add_layer(self, number_of_entries, number_of_neurons, activation_function):
        self.layers.append(Layer(number_of_entries, number_of_neurons, activation_function, self.optim))

    def MSE(self, x, y):
        return np.mean(np.square(x - y))

    def deriv_MSE(self, x, y):
        return 2 * (x - y)

    def predict(self, x):
        x_ = np.array([x])
        for layer in self.layers:
            x_ = layer.forward(x_)
        return x_

    def train(self, x, y, eta):
        y = np.array([y])
        x_ = self.predict(x)
        error = self.MSE(x_, y)
        delta_l_1 = self.layers[-1].deriv_act_fun(self.layers[-1].layer_before_activation[0]) * self.deriv_MSE(x_, y)
        for i in range(1, len(self.layers)):
            delta_l_1 = self.layers[-i].backward(self.layers[-i-1], delta_l_1, eta)
        self.layers[0].backward_first_layer(np.array([x]), delta_l_1, eta)
        return error

    def __repr__(self):
        layers = ""
        if (len(self.layers) > 0):
            layers += "Number of entries: " + str(self.layers[0].weights.shape[0]) + "\n\n"
        for layer in self.layers:
            layers += "Layer N°" + str(self.layers.index(layer) + 1) + ": " + str(layer.weights.shape[1]) + " neurons,  Activation Function: "
            if layer.activation_function != None:
                layers += " " + layer.activation_function + "\n"
            else:
                layers += " " + "No function found" + "\n"
        return layers

    def __len__(self):
        return len(self.layers)





## Quelques programmes auxiliaires nécessaires pour la sauvegarde du jeu

def open_variable(name):
    filename = name + ".pickle"
    with open(filename, "rb") as f:
        variable = pickle.load(f)
        return variable


def save_variable(name, variable):
    filename = name + ".pickle"
    with open(filename, "wb") as f:
        pickle.dump(variable, f)
        f.close()

def val_to_list(dict):
    return list(dict.values())

def key_to_list(dict):
    return list(dict.keys())


def copy(fichier, destination):
    copyfile(fichier, destination)

##  JEU ##

#copy(emplacement_permament, base_de_donnees)

dernierHid = 0
def affiche_console(historique):
    global dernierHid
    for i in range(dernierHid, len(historique)):
        print(*historique[i][:-1])
    dernierHid = len(historique)

from itertools import permutations, product

def verifie_draft(choixJoueur, listesChoix):
    for possibilites in product(*listesChoix):
        if tuple(choixJoueur) in permutations(possibilites):
            return True
    return False

## STRATEGIE IA

cupt = 1
temperature = 1
root = None
noeud = root

dicPidEmplA = {}
dicPidEmplE = {}

def draft(listesChoix):
    L = []
    for x in listesChoix:
        L.append(x[randint(0, 1)])
    return L


def _pid_empl_relation(etatJeu):

    """ creer un dictionnaire qui associe le pid à l'emplacement """

    global dicPidEmpl
    for p in etatJeu.equipes[0]:
        if p != None:
            dicPidEmplA[p.pid] = etatJeu.equipes[0].index(p)

    for p in etatJeu.equipes[1]:
        if p != None:
            dicPidEmplE[p.pid] = etatJeu.equipes[1].index(p)



def _get_possible_actions_for_actual_state(etatDuJeu):
    """
    On renvoie une liste de taille 768 (la taille de toutes les actions possibles avec des 0 aux indices qui correspondent à une action qu'on ne peut pas prendre et des 1 là où on peut prendre une action

    """
    actions_filter = [0] * 75
    actions = [None] * 75
    s = 0

    for personnageA in etatDuJeu.equipes[0]:
        for personnageE in etatDuJeu.equipes[1]:
            if personnageA != None and personnageE != None:
                for capacite in range(3):
                    if personnageA.capacites[capacite].attente == 0:
                        action = (personnageA.pid, personnageE.pid, capacite)
                        actions_filter[s] = 1
                        actions[s] = action
                        s += 1

    return actions, actions_filter




########################################################



""" Programmes pour obtenir l'etat de jeu """


def _normalizer(state):
    somme = sum(state)
    state = [i / somme for i in state]
    return state


def _get_game_state_not_normalized(etatJeu):

    """ Renvoyer une liste contenant : le joueur qui doit jouer + les personnages + les caracteristiques de chaque personnage """

    state = [0] * 51
    if etatJeu.doitJouer == None:
        personnageQuiJoue = 0
    else:
        personnageQuiJoue = etatJeu.doitJouer.pid
    state[0] = personnageQuiJoue
    s = 1
    L = [1, 3, 5, 7, 9]

    A = etatJeu.equipes[0]; E = etatJeu.equipes[1]
    for personnage in A:
        if personnage != None:
            state[  s  ] = personnage.pid
            state[s + 1] = personnage.vie
            state[s + 2] = personnage.force
            state[s + 3] = personnage.vitesse
            state[s + 4] = personnage.esquive
        if A.index(personnage) in L:
            s += 5

    for personnage in E:
        if personnage != None:
            state[  s  ] = personnage.pid
            state[s + 1] = personnage.vie
            state[s + 2] = personnage.force
            state[s + 3] = personnage.vitesse
            state[s + 4] = personnage.esquive
        if E.index(personnage) in L:
            s += 5
    return state

def _get_game_state(etatJeu):
    state = _get_game_state_not_normalized(etatJeu)
    return _normalizer(state)



#####################################################################"

""" programmes qui développent l'arbre de recherche """


def _initialise_root(etatDuJeu):
    global root, noeud
    root = Noeud(None, None, etatDuJeu.doitJouer.equipe)
    _expand_noeud(root, etatDuJeu, network_to_train)
    root.Ns = 1
    noeud = root


def _expand_noeud(noeud, etatDuJeu, network):
    global possible_actions
    actions, actions_filter = _get_possible_actions_for_actual_state(etatDuJeu)
    state = _get_game_state(etatDuJeu)
    prediction = network.predict(state)
    Ps, v = prediction[PROBABILITIES], prediction[VALUE]
    for i in range(len(actions)):
        if actions[i] != None:
            noeud.ajouter_enfant(actions[i], Ps[0][i], i)
    return v[0][0]

def _faire_un_back_up(n, v):
    previous = n.action
    n = n.parent
    while n != None:
        c_a = n.edges[previous]
        n.Ns -= c_a["N_s_a"] ** (1/temperature)
        c_a["N_s_a"] += 1
        c_a["W_s_a"] += v
        c_a["Q_s_a"] =  c_a["W_s_a"]/c_a["N_s_a"]
        n.Ns += c_a["N_s_a"] ** (1/temperature)
        n.N += 1
        previous = n.action
        n = n.parent


##########################################################

def lance_jeu_une_fois(etatJeu, action):
    j = etatJeu.doitJouer.equipe

    pidA, pidE, i = action[0], action[1], action[2]

    cibleAdverse, cibleAlliee = dicPidEmplE[pidE], dicPidEmplA[pidA]


    etatJeu.change_cible_adverse(coords(cibleAdverse), j)
    etatJeu.change_cible_alliee(coords(cibleAlliee), j)


    if etatJeu.doitJouer.capacites[i].attente != 0:
        i = 0

    etatJeu.applique_capacite(etatJeu.doitJouer.capacites[i], etatJeu.doitJouer)

    etatJeu.fin_de_tour()


    return etatJeu


def lance_jeu(etatInitial, done):
    if etatInitial != None:
        etatJeu = etatInitial
        _pid_empl_relation(etatJeu)
    else:
        etatJeu = EtatJeu()

        dernierPersonnage = list(curseur.execute("SELECT MAX(pid) FROM capacites WHERE EXISTS (SELECT 1 FROM occurences WHERE occurences.cid = capacites.cid)"))[0][0]

        listesChoix = [[1+randrange(dernierPersonnage) for j in range(2)] for i in range(5)]

        for j in range(2):
            try:
                choixJoueur = draft(listesChoix)
            except:
                print("Erreur à l'exécution du draft pour le joueur ", j)
                choixJoueur = [liste[0] for liste in listesChoix]

            # Draft incorrect, on remplace par le premier choix de chaque liste de choix
            if not verifie_draft(choixJoueur, listesChoix):
                print("Draft incorrect pour le joueur ", j)
                choixJoueur = [liste[0] for liste in listesChoix]
            etatJeu.equipes[j] = initialise_equipe([None, choixJoueur[1], None, choixJoueur[3], None, choixJoueur[0], None, choixJoueur[2], None, choixJoueur[4]], j)


        _pid_empl_relation(etatJeu)
        save_variable("etatJeu", etatJeu)

    memo = None
    s = 0
    while True:
        resultat = etatJeu.debut_de_tour()
        if not done:
            _initialise_root(etatJeu)
            done = True

        if resultat != PRET_AU_COMBAT:
            if memo != None:
                v = _points(etatJeu)
                etat = _get_game_state_not_normalized(etatJeu)
                noeud.enfants[(tuple(etat), memo)] = (Noeud(noeud, memo, None), etatJeu)
                _faire_un_back_up(noeud, v)
            return

        a, b, c, memo = tour_de_jeu(etatJeu, memo, etatJeu.doitJouer.equipe, s)

        s = 1
        action = a, b, c

        if memo == FIN_DU_JEU:
            return


        etatJeu = lance_jeu_une_fois(etatJeu, action)



def tour_de_jeu(etatDuJeu, memo, j, s):

    """ On applique MCTS en même temps que le jeu (c'est plus pratique)"""

    global root, network_to_train, noeud, previous_act
    global possible_actions


    etat = _get_game_state_not_normalized(etatDuJeu)

    if s != 0 :
        if (tuple(etat), previous_act) in noeud.enfants:
            noeud = noeud.enfants[(tuple(etat), previous_act)][0]
        else:
            noeud.enfants[(tuple(etat), previous_act)] = (Noeud(noeud, previous_act, j), etatDuJeu)
            noeud = noeud.enfants[(tuple(etat), previous_act)][0]

            v = _expand_noeud(noeud, etatDuJeu, network_to_train)
            if j == 0:
                _faire_un_back_up(noeud, v)
            else:
                _faire_un_back_up(noeud, -v)
            noeud = root

            return None, None, None, FIN_DU_JEU


    max_u = -float("inf")
    best_action = -1

    if s == 0:
        epsilon = 0.2
        nu = np.random.dirichlet([0.8] * len(noeud.edges))
    else:
        epsilon = 0
        nu = [0] * len(noeud.edges)

    for (action, c) in list(noeud.edges.items()):
        P_a = c["P_s_a"]; Q_s_a = c["Q_s_a"]; N_s_a = c["N_s_a"]; N = noeud.N

        idx = list(noeud.edges.keys()).index(action)
        U = cupt * ((1-epsilon) * P_a + epsilon * nu[idx]) * (N** 0.5 ) / (1+N_s_a)

        if Q_s_a + U > max_u:

            max_u = Q_s_a + U
            best_action = action

    previous_act = best_action

    return best_action[0], best_action[1], best_action[2], best_action



########################################### RECHERCHE DE L'ARBRE

def _is_game_finished(et):
    for pa in et.equipes[0]:
        for pe in et.equipes[1]:
            if pa != None and pe != None:
                return False
    return True




def _get_etatJeu_from_enfants(node, action):
    for ((s, act), enfant) in list(node.enfants.items()):
        if action == act and enfant[0].action == action:
            return s, enfant[1]

    return None, None



def _get_training_examples(nr):
    global root, noeud
    training = []
    lance_jeu(None, False)
    while True:
        for i in range(nr):
            lance_jeu(open_variable("etatJeu"), True)

        etat = open_variable("etatJeu")
        probabilities = _get_action_probabilities(root)
        etat.debut_de_tour()
        training.append([_get_game_state(etat), probabilities, None])

        action = _choose_action_for_node(root)
        state, etatJeu = _get_etatJeu_from_enfants(root, action)

        root = root.enfants[(tuple(state), action)][0]
        root.parent = None
        root.action = None
        noeud = root

        save_variable("etatJeu", etatJeu)

        resultat = etatJeu.debut_de_tour()

        if resultat != PRET_AU_COMBAT:
            _set_gain(training, _points(etatJeu))
            return training




def _points(etatDuJeu):

    """ Renvoyer 1 si j'ai gagné, -1 sinon """

    for i in etatDuJeu.equipes[0]:
        if i != None:
            return 1
    return -1

def _get_action_probabilities(node):
    P = [0] * 75
    N = node.N
    if not node.est_feuille():
        for (action, c) in list(node.edges.items()):
            P[c["index"]]=c["N_s_a"] / N if N > 0 else 0
        return P
    else:
        return None

def _choose_action_for_node(node):
    P = _get_action_probabilities(node)
    if P == None:
        return P
    index = P.index(max(P))
    return node.index_to_action[index]

def _set_gain(training, v):
    for example in training:
        example[2] = v






## GENERER DES DONNEES

def generate_data(n, nr):
    donnees = []
    print("\n\n\n Generating Data ... \n\n\n")
    print("============================ Process Started ============================")
    for i in range(n):
        data = _get_training_examples(nr)
        donnees += data

    save_variable("donnees", donnees)
    print(".\n. Length of Data:", len(donnees), "\n.\n============================ Process Finished ============================")
    save_variable("neural_network", network_to_train)




## FAIRE COMBATRE LES IA

def tour_de_jeu_ia(etatJeu, network, j):
    actions, actions_filter = _get_possible_actions_for_actual_state(etatJeu)
    state = _get_game_state(etatJeu)
    prediction = network.predict(state)
    Ps, v = prediction[PROBABILITIES], prediction[VALUE]
    Ps = Ps * np.array([actions_filter])
    Ps = list(Ps[0])

    idx = Ps.index(max(Ps))


    return actions[idx]




def lance_jeu_ia(joueur1, joueur2):
    joueurs = [joueur1, joueur2]

    etatJeu = EtatJeu()

    dernierPersonnage = list(curseur.execute("SELECT MAX(pid) FROM capacites WHERE EXISTS (SELECT 1 FROM occurences WHERE occurences.cid = capacites.cid)"))[0][0]

    listesChoix = [[1+randrange(dernierPersonnage) for j in range(2)] for i in range(5)]

    for j in range(2):
        try:
            choixJoueur = draft(listesChoix)
        except:
            print("Erreur à l'exécution du draft pour le joueur ", j)
            choixJoueur = [liste[0] for liste in listesChoix]

        # Draft incorrect, on remplace par le premier choix de chaque liste de choix
        if not verifie_draft(choixJoueur, listesChoix):
            print("Draft incorrect pour le joueur ", j)
            choixJoueur = [liste[0] for liste in listesChoix]
        etatJeu.equipes[j] = initialise_equipe([None, choixJoueur[1], None, choixJoueur[3], None, choixJoueur[0], None, choixJoueur[2], None, choixJoueur[4]], j)

        _pid_empl_relation(etatJeu)

    while True:
        resultat = etatJeu.debut_de_tour()
        if resultat != PRET_AU_COMBAT:
            return etatJeu
        j = etatJeu.doitJouer.equipe
        #try:
        if True:

            pidA, pidE, i = tour_de_jeu_ia(deepcopy(etatJeu), joueurs[j], j)
            cibleAdverse, cibleAlliee = dicPidEmplE[pidE], dicPidEmplA[pidA]
        #except:
            #print("Erreur à l'exécution du tour de jeu du joueur", j)
            #exit()
        etatJeu.change_cible_adverse(coords(cibleAdverse), j)
        etatJeu.change_cible_alliee(coords(cibleAlliee), j)


        if etatJeu.doitJouer.capacites[i].attente != 0:
            i = 0

        etatJeu.applique_capacite(etatJeu.doitJouer.capacites[i], etatJeu.doitJouer)

        etatJeu.fin_de_tour()




def transform(donnee):
    for d in donnee:
        P = [0] * 75
        index = d[1].index(max(d[1]))
        P[index] = 1
        d[1] = P
    return donnee



##Entrainer l'IA

def train_(net, donnee):
    prob = net.reseau_probas
    val = net.reseau_values

    for entrainement in range(50):
        err = 0
        for d in donnee:
            err += prob.train(d[0], d[1], 0.001)
            val.train(d[0], d[2], 0.001)
        print(err/50)


def choose_best_ia(old_network, trained_network):
    """ Choisir la meilleure IA """

    etatJeu = lance_jeu_ia(trained_network, old_network)
    p = _points(etatJeu)
    return int(p != - 1)


def winning_rate():
    w = 0
    total = 0
    old = open_variable("neural_network")
    new = open_variable("trained_network")
    for _ in range(20):
        b = choose_best_ia(old, new)
        w += b
        total += 1

    return w/total


## Enfin le programme final, tourner celui-ci pour voir la magie apparaître !


def training_loop(n, exists, nr_de_recherche_MCTS):
    global network_to_train
    if not exists:
        network_to_train = NeuralNetworks(51, 75, optimizer = "Adam")
    else:
        network_to_train = open_variable("trained_network")
    save_variable("init_net", network_to_train)
    for i in range(n):
        print("\n\n\n GENERATION N°: ", i+1)
        generate_data(10, nr_de_recherche_MCTS)
        donnees = transform(open_variable("donnees"))
        shuffle(donnees)

        print("\n\n\n ====================== Starting Training ======================")
        train_(network_to_train, donnees)
        save_variable("trained_network", network_to_train)
        w = winning_rate()
        if w >= 0.5:
            network_to_train = open_variable("trained_network")
        else:
            network_to_train = open_variable("neural_network")


            
            
            
            
 training_loop(10, False, 800)
