import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from strlearn.metrics import recall, specificity, f1_score, geometric_mean_score_1, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from tabulate import tabulate
from scipy.stats import ttest_ind
from deslib.des import KNORAU, KNORAE, DESKNN
from sklearn.ensemble import AdaBoostClassifier

clfs = {
    "DES_KNN": DESKNN(random_state=997),
    "KNORA-U": KNORAU(random_state=997),
    "KNORA-E": KNORAE(random_state=997),
    "ADABoost": AdaBoostClassifier(random_state=997),
}

datasets = [
    'ecoli4',
    'ecoli-0-1-4-6_vs_5',
    'glass2',
    'glass5',
    'glass6',
    'glass-0-1-6_vs_2',
    'glass-0-1-6_vs_5',
    'new-thyroid1',
    'newthyroid2',
    'yeast4',
    'yeast6',
    'yeast-1-2-8-9_vs_7',
    'yeast-2_vs_4',
    'yeast-1_vs_7',
    'pima',
    'segment0',
    'vehicle0',
    'page-blocks-1-3_vs_4',
    'vowel0',
    'wisconsin',
]

preprocs = {
    'none': None,
    'ROS': RandomOverSampler(random_state=3721),
    'SMOTE' : SMOTE(random_state=3721),
    'ADASYN': ADASYN(random_state=3721),
}

metrics = {
    'Recall': recall,
    'Specificity': specificity,
    'F1': f1_score,
    'G-mean': geometric_mean_score_1,
    'BAC': balanced_accuracy_score,
}

#testujemy per klasyfikator zespołowy
for clf_id, clf_name in enumerate(clfs):

    clf = clfs[clf_name]
    print('\n\nTesty dla',clf_name)

    # wielokrotna stratyfikowana walidacja krzyżowa - podczas podziału na podzbiory zachowuje oryginalny lub zbliżony do oryginalnego poziom niezbalansowania
    n_splits = 5
    n_repeats = 2
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=3721)

    # Przygotowujemy tablicę wyników wypełniając ją zerami
    scores = np.zeros((len(preprocs),len(datasets), n_splits * n_repeats, len(metrics)))

    # wyliczanie jakości per dataset
    for data_id, dataset in enumerate(datasets):

        # importowanie danych
        dataset = np.genfromtxt("datasets/%s.csv" % (dataset), delimiter=",")
        # dzielenie datasetów na X i y
        # gdzie do X przypisujemy wszystkie kolumny oprócz ostatniej, do y ostatnią i rzutujemy ją na liczbę całkowitą (i tak jest całkowita)
        X = dataset[:, :-1]
        y = dataset[:, -1].astype(int)

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            for preproc_id, preproc in enumerate(preprocs):

                clf = clone(clf)

                # preprocessing danych
                if preprocs[preproc] == None:
                    X_train, y_train = X[train], y[train]
                else:
                    X_train, y_train = preprocs[preproc].fit_resample(X[train], y[train])

                # dopasowanie modelu do danych uczących
                clf.fit(X_train, y_train)

                # predykcja
                y_pred = clf.predict(X[test])

                # ocena jakości w różnych metrykach
                scores[preproc_id, data_id, fold_id, 0] = recall(y[test], y_pred)
                scores[preproc_id, data_id, fold_id, 1] = specificity(y[test], y_pred)
                scores[preproc_id, data_id, fold_id, 2] = f1_score(y[test], y_pred)
                scores[preproc_id, data_id, fold_id, 3] = geometric_mean_score_1(y[test], y_pred)
                scores[preproc_id, data_id, fold_id, 4] = balanced_accuracy_score(y[test], y_pred)



    #########################################################################
    # ANALIZA STATYSTYCZNA
    #########################################################################

    # Czyszczenie pliku wynikowego ze starych wyników (otwieranie w trybie 'w' i zamykanie)
    file = open('./results/results-'+clf_name+'.txt', 'w').close()

    # Aby zapisywać wyniki do pliku
    with open('./results/results-'+clf_name+'.txt', 'a') as file:

        for score_id, score_name in enumerate(metrics):

            # dwuwymiarowe tablice przygotowane dla t-statystyki i p-wartośći, wartość alpha
            t_statistic = np.zeros((len(preprocs), len(preprocs)))
            p_value = np.zeros((len(preprocs), len(preprocs)))
            alfa = 0.05

            for dataset_id in range(len(datasets)):
                scores_F = scores[:, dataset_id, :, score_id]
                for i in range(len(preprocs)):
                    for j in range(len(preprocs)):
                        t_statistic[i, j], p_value[i, j] = ttest_ind(scores_F[i], scores_F[j])

                advantage = np.zeros((len(preprocs), len(preprocs)))
                advantage[t_statistic > 0] = 1

                significance = np.zeros((len(preprocs), len(preprocs)))
                significance[p_value <= alfa] = 1

                sign_better = significance * advantage
                
                headers = list(preprocs.keys())
                names_column = np.expand_dims(headers, axis=1)
                sign_better_table = tabulate(np.concatenate((names_column, sign_better), axis=1), headers)
                file.write(f"\n\nStatystycznie znaczaco lepszy od: ({datasets[dataset_id]} dla {clf_name} dla metryki {score_name})\n{sign_better_table}\n")
            
            file.write(f"\n\n\n\n\n")



    #########################################################################
    # WYKRESY
    #########################################################################

    # scores mają rozmiar: 4x20x10x5 (4 metody / 20 datasetów / 10 foldów / 5 metryk)
    # uśredniamy po datasetach (drugi wymiar)
    mean_scores_chart = np.mean(scores, axis=1)
    # ponownie uśredniamy, tym razem po foldach (ponownie drugi wymiar) i transponujemy
    mean_scores_chart = np.mean(mean_scores_chart, axis=1).T
    # po uśrednieniu scores mają rozmiar 5x4 (5 metryk, 4 metody)

    # metryki i metody
    chart_metrics=["Recall", 'Specificity', 'F1', 'G-mean', 'BAC']
    chart_methods=["None", 'ROS', 'SMOTE', 'ADASYN']
    N = mean_scores_chart.shape[0]

    # kąt dla każdej z osi
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # spider plot
    ax = plt.subplot(111, polar=True)

    # pierwsza oś na górze
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # po jednej osi na metryke
    plt.xticks(angles[:-1], chart_metrics)

    # oś y
    ax.set_rlabel_position(0)
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    ["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"],
    color="grey", size=7)
    plt.ylim(0,1)

    # Dodajemy wlasciwe ploty dla każdej z metod
    for method_id, method in enumerate(chart_methods):
        values=mean_scores_chart[:, method_id].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=method)

    # Dodajemy legendę
    plt.legend(bbox_to_anchor=(1.15, -0.05), ncol=5)

    # Zapisujemy wykres
    plt.savefig('./results/chart-'+clf_name+'.png', dpi=200)

    # Resetujemy plot aby wyniki per clf się nie nakładały
    plt.clf()
    plt.cla()



    #########################################################################
    # UŚREDNIONE WYNIKI PER DATASET
    #########################################################################

    # # scores mają rozmiar: 4x20x10x5 (4 metody / 20 datasetów / 10 foldów / 5 metryk)
    # # uśredniamy po foldach (trzeci wymiar) i transponujemy
    # mean_scores_stat = np.mean(scores, axis=2).T
    # # po uśrednieniu scores mają rozmiar 5x20x4 (5 metryk, 20 datasetów, 4 metody)

    # scores_per_metric = {
    #     'Recall':       mean_scores_stat[0],
    #     'Specificity':  mean_scores_stat[1],
    #     'F1':           mean_scores_stat[2],
    #     'G-mean':       mean_scores_stat[3],
    #     'BAC':          mean_scores_stat[4],
    # }

    # # etykiety kolumn i wierszy (szczegółowe dane)
    # headers = list(preprocs.keys())
    # names_column = np.expand_dims(datasets, axis=1)
    # # prezentacja szczegółowych danych dla metryki
    # scores_M = np.concatenate((names_column, scores_per_metric[score_name]), axis=1)
    # scores_M = tabulate(scores_M, headers, tablefmt="2f", floatfmt='0.3f')
    # print(f"Usrednionie wyniki dla metryki {score_name}\n{scores_M}\n")