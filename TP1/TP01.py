import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import Imputer

n_base = 100

data1 = np.random.randn(n_base, 2) + [5, 5]
data2 = np.random.randn(n_base, 2) + [3, 2]
data3 = np.random.randn(n_base, 2) + [1, 5]
data = data1 + data2 + data3
data = np.concatenate((data1, data2, data3))  # to concatenate lists or series
print(data.shape)  # vérification
np.random.shuffle(data)
n_samples = data.shape[0]
# visualisation (optionnelle) des données générées
print("================", "n_samples", n_samples, sep="\n")

# plt.plot(data[:, 0], data[:, 1], 'r+')
# plt.show()
missing_rate = 0.3  # taux de lignes à valeurs manquantes
n_missing_samples = int(np.floor(n_samples * missing_rate))
print("================", "n_missing_samples", n_missing_samples, sep="\n")

# choix lignes à valeurs manquantes
missing_samples = np.hstack(
    (
        np.zeros(n_samples - n_missing_samples, dtype=np.bool),
        np.ones(n_missing_samples, dtype=np.bool)
    )
)

np.random.shuffle(missing_samples)
print("================", "missing_samples", missing_samples, sep="\n")
# obtenir la matrice avec données manquantes : manque indiqué par
# valeurs NaN dans la seconde colonne pour les lignes True dans
# missing_samples
data_missing = data.copy()
data_missing[np.where(missing_samples), 1] = np.NAN
print("================", "data_missing", data_missing, sep="\n")
print("================", "data_missing type", type(data_missing), sep="\n")
data_missing = pd.DataFrame(data_missing)
deldata = data_missing.dropna()

print("================", "dimension data", deldata.shape, sep="\n")

# imputation par la moyenne
imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
data_imputed = imp.fit_transform(data_missing)

print("================", "data_imputed", data_imputed, sep="\n")
# calculer l'"erreur" d'imputation
mean_squared_error = mean_squared_error(data[missing_samples, 1], data_imputed[missing_samples, 1])
print("================", "error d'imputation", mean_squared_error, sep="\n")
