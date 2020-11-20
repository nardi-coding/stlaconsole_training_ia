import numpy as np
class NeuralNetworks:
    def __init__(self, nr_entrees, nr_d_actions, optimizer = "SDG"):
        self.reseau_probas = Network(optimizer = optimizer)
        self.reseau_values = Network(optimizer = optimizer)

        self.reseau_probas.add_layer(nr_entrees, 32, "relu")
        self.reseau_probas.add_layer(32, 64, "relu")
        self.reseau_probas.add_layer(64, 32, "relu")
        self.reseau_probas.add_layer(32, 64, "relu")
        self.reseau_probas.add_layer(64, nr_d_actions, "softmax")

        self.reseau_values.add_layer(nr_entrees, 64, "sigmoid")
        self.reseau_values.add_layer(64, 32, "sigmoid")
        self.reseau_values.add_layer(32, 1, "tanh")

        self.error = 0


    def _get_error(self):
        return self.error

    def MSE(self, x, y):
        return np.mean(np.square(x - y))

    def deriv_MSE(self, x, y):
        return 2 * (x - y)


    def alphago_loss_function(self, pi_p, pi_vrai, v_p, v_vrai):
        mse = self.MSE(v_p, np.array([v_vrai]))
        log = -np.mean(np.array([pi_vrai]) * np.log(pi_p + 1e-8))
        return mse + log

    def deriv_alpha_go(self, pi_p, pi_vrai, v_p, v_vrai):
        d_mse = self.deriv_MSE(v_p, v_vrai)
        d_log = pi_p - pi_vrai
        return d_mse, d_log

    def forward(self, x, pi, v):
        vecteur_probas, valeur_de_gain = self.predict(x)
        self.error = self.alphago_loss_function(vecteur_probas, pi, valeur_de_gain, v)
        return self.deriv_alpha_go(vecteur_probas, pi, valeur_de_gain, v)

    def train(self, x, pi, v, eta):
        d_mse, d_log = self.forward(x, pi, v)

        self.reseau_values.train(x, eta, d_mse, False)
        self.reseau_probas.train(x, eta, d_log, True)
        return self.error


    def predict(self, entree):
        vecteur_probas = self.reseau_probas.predict(entree)
        valeur_de_gain = self.reseau_values.predict(entree)
        return vecteur_probas, valeur_de_gain

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
        elif self.activation_function == "softmax":
            Z_ = Z - np.max(Z)
            exp_ = np.exp(Z_)
            return exp_ / exp_.sum()
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

    def predict(self, x):
        x_ = np.array([x])
        for layer in self.layers:
            x_ = layer.forward(x_)
        return x_

    def train(self, x, eta, deriv_err, soft = False):
        if not soft:
            delta_l_1 = self.layers[-1].deriv_act_fun(self.layers[-1].layer_before_activation[0]) * deriv_err
        else:
            delta_l_1 = deriv_err
        for i in range(1, len(self.layers)):
            delta_l_1 = self.layers[-i].backward(self.layers[-i-1], delta_l_1, eta)
        self.layers[0].backward_first_layer(np.array([x]), delta_l_1, eta)

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
