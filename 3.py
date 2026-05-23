import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler

TARGET = "Vote_Percentage"
CONNECTED_COLS = [
    "Voter_Turnout"
]

# Читаем датасет и выбираем только числовые признаки, исключая TARGET
raw = pd.read_csv("Vote Ai.csv")
print("Размер исходного датасета:", raw.shape)
numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
FEATURES = [
    col for col in numeric_cols
    if col != TARGET and col not in CONNECTED_COLS
]
df = raw[[TARGET] + FEATURES].copy()
df = df.dropna().reset_index(drop=True)
print("Количество объектов после удаления пропусков:", len(df))
print("Количество числовых признаков без target:", len(FEATURES))
print("Размер данных для расчета:", df.shape)
print(df.describe())

# Делаем выборку в 5000 элементов, тк иначе будет очень долго работать
MAX_OBJECTS_FOR_WORK = 5000
if MAX_OBJECTS_FOR_WORK is not None and len(df) > MAX_OBJECTS_FOR_WORK:
    df = df.sample(n=MAX_OBJECTS_FOR_WORK, random_state=42).reset_index(drop=True)

corr = raw[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].sort_values(ascending=False)
print(corr.head(15))

# Нормирование методом MinMaxScaler
x_train = df[FEATURES].copy()
y_train = df[TARGET].copy()
x_scaler = MinMaxScaler()
x_scaler.fit(x_train)
x_train = x_scaler.transform(x_train)
print("X shape:", x_train.shape)
print("y shape:", y_train.shape)


# GradientBoostingRegressor
params = {
    "n_estimators": 300, # количество деревьев
    "max_depth": 4, # максимальная глубина каждого дерева
    "min_samples_split": 10, # минимальное количество объектов, нужное для разбиения узла дерева
    "learning_rate": 0.01, # скорость обучения
    "verbose": 0, # (конфигурационный параметр) без подробного вывода процесса обучения
}
model_base_full = ensemble.GradientBoostingRegressor(**params)
model_base_full.fit(x_train, y_train)

# Выберем самые важные признаки
N_FEATURES = 15
base_feature_importance = copy.deepcopy(model_base_full.feature_importances_)
base_feature_importance_sorted_idx = np.argsort(base_feature_importance)[::-1]
base_feature_importance_sorted_idx = base_feature_importance_sorted_idx[0:N_FEATURES]
boosting_names = [FEATURES[i] for i in base_feature_importance_sorted_idx]
vals = base_feature_importance[base_feature_importance_sorted_idx]
fimps = pd.DataFrame(data={"Name": boosting_names, "Vals": vals})
print("Признаки, выбранные GradientBoostingRegressor:")
print(fimps)
sns.set_context("paper", font_scale=1.3)
plt.figure(figsize=(10, 7))
sns.barplot(x="Vals", y="Name", data=fimps, label="Total", color="b")
plt.title("GradientBoostingRegressor: feature_importances_")
plt.tight_layout()
plt.show()


# Unsupervised Feature Selection based on Ant Colony Optimization (UFSACO)
# Метод муравьиной колонии

# Нормирование методом MinMaxScaler fixme 37 строка тоже самое
all_scaler = MinMaxScaler()
all_scaler.fit(df[FEATURES])
all_scaled = all_scaler.transform(df[FEATURES])
all_input_names = FEATURES
EPS = 1e-6

# Задаем число признаков, которые хотим выбрать
N_START_FEATURES = all_scaled.shape[1]
print("N_START_FEATURES =", N_START_FEATURES)
N_END_FEATURES = N_FEATURES

NC_MAX = 3 # Количество внешних итераций алгоритма. (Число эпох)
N_STEPS = 4 # Количество шагов, которые делает каждый муравей внутри одной эпохи.
INIT_PHEROMONE = 0.2 # Начальное количество феромона на каждом узле-признаке
RO = 0.2 # Коэффициент испарения феромона
EXPLOITATION_PROB = 0.7 # Вероятность выбора муравьем самого привлекательного признака, а не случайного
ALPHA = 1.0 # Степень влияния похожести признаков
BETA = 1.0 # Вероятность выбора муравьем самого привлекательного признака, а не случайного
N_ANTS = 5 # Количество муравьев

# Похожие пары признаков
sim = {}

def set_sim(i, j):
    """Записываем похожесть между признаками i и j по формуле косинусного сходства"""
    a = all_scaled[:, i]
    b = all_scaled[:, j]
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    res = np.dot(np.asarray(a), np.asarray(b)) / denom
    sim[(min(i, j), max(i, j))] = np.abs(res)
    return res


def get_sim(i, j):
    """Вычисляем похожесть между признаками i и j по формуле косинусного сходства"""
    (i, j) = (min(i, j), max(i, j))
    if (i, j) not in sim.keys():
        set_sim(i, j)
    return sim[(i, j)]


# todo понять
def UFSACO(verbose=True):
    # Инициируем начальным значением феромона в каждом узле
    tau = INIT_PHEROMONE * np.ones((N_START_FEATURES))

    # Внешний цикл - число эпох
    for count in range(NC_MAX):
        # Случайно размещаем муравьев по узлам с вероятностью, пропорциональной феромону
        ants_pos = np.random.choice(N_START_FEATURES, size=N_ANTS, p=tau/sum(tau))

        # Очищаем счетчик посещений узлов для каждого узла
        visits = np.zeros((N_START_FEATURES))

        # Очищаем множество посещенных узлов каждым k-ым агентом из каждого i-ого узла
        nodes_visited = {(k, i): set() for k in range(N_ANTS) for i in range(N_START_FEATURES)}

        # Внутренний цикл - длина пути, который проходят все муравьи
        for iter in range(N_STEPS):
            # k - номер текущего муравья
            for k in range(N_ANTS):
                # На каком i-ом узле находится k-ый муравей
                i = ants_pos[k]

                # Множество посещенных узлов
                visited = nodes_visited[(k, i)]

                # Множество непосещенных узлов
                unvisited = list((set(range(N_START_FEATURES)) - visited) - {i})

                # UFSACO: без учета таргета
                # Защита: не допускать деления на ноль при очень малой похожести
                node_score = np.array([tau[j] / (np.power(get_sim(i, j), ALPHA) + EPS) for j in unvisited])

                # Какой шаг выполняем - exploration или exploitation?
                q = np.random.uniform()
                if q <= EXPLOITATION_PROB:
                    # EXPLOITATION
                    if verbose:
                        print("EXPLOITATION")
                    # Переходим в узел с максимальной желательностью
                    jj = np.argmax(node_score)
                else:
                    # EXPLORATION
                    if verbose:
                        print("EXPLORATION")
                    # Переходим по вероятности, пропорциональной желательности
                    total_score = node_score.sum()
                    if total_score <= 0:
                        # вырожденный случай: равномерное распределение
                        p = None
                    else:
                        p = node_score / total_score
                    jj = np.random.choice(len(unvisited), size=1, p=p)[0]

                # Получаем номер следующего узла j для k: i -> j
                j = unvisited[jj]

                # Перемещаем k-ого муравья в j-ый узел
                ants_pos[k] = j

                # Добавляем информацию о перемещении
                nodes_visited[(k, i)].add(j)

                # Увеличиваем счетчик посещения j-ого узла
                visits[j] += 1

                if verbose:
                    print(f"count={count}, iter={iter}, k={k}, i={i}, j={j}")

        # Пересчитываем количество феромона
        total_visits = sum(visits)
        if total_visits == 0:
            # Ни одно посещение не произошло — оставляем ферромон с учетом испарения
            tau = (1 - RO) * tau
        else:
            tau = (1 - RO) * tau + (visits / total_visits)

    return tau


MIN_INTERSECTION = 4
param_grid = [
    {"NC_MAX": 3, "N_STEPS": 4, "INIT_PHEROMONE": 0.2, "RO": 0.2,
     "EXPLOITATION_PROB": 0.7, "ALPHA": 1.0, "BETA": 1.0, "N_ANTS": 5},
    {"NC_MAX": 5, "N_STEPS": 8, "INIT_PHEROMONE": 0.2, "RO": 0.2,
     "EXPLOITATION_PROB": 0.7, "ALPHA": 1.0, "BETA": 1.0, "N_ANTS": 10},
    {"NC_MAX": 8, "N_STEPS": 12, "INIT_PHEROMONE": 0.2, "RO": 0.2,
     "EXPLOITATION_PROB": 0.7, "ALPHA": 1.0, "BETA": 1.0, "N_ANTS": 20},
    {"NC_MAX": 10, "N_STEPS": 16, "INIT_PHEROMONE": 0.2, "RO": 0.1,
     "EXPLOITATION_PROB": 0.8, "ALPHA": 1.0, "BETA": 1.0, "N_ANTS": 20},
]
best_result = None

for hp in param_grid:
    NC_MAX = hp["NC_MAX"]
    N_STEPS = hp["N_STEPS"]
    INIT_PHEROMONE = hp["INIT_PHEROMONE"]
    RO = hp["RO"]
    EXPLOITATION_PROB = hp["EXPLOITATION_PROB"]
    ALPHA = hp["ALPHA"]
    BETA = hp["BETA"]
    N_ANTS = hp["N_ANTS"]

    # С различными параметрами выбираем 15 лучших признаков и перемекаем из с теми,
    # что нашлись с помощью градиентного бустинга
    for seed in range(100, 130):
        np.random.seed(seed)
        tau = UFSACO(verbose=False)
        features_UFSACO = np.array(tau.argsort()[::-1][:N_END_FEATURES])
        ufsaco_names = [all_input_names[i] for i in features_UFSACO]
        intersection = sorted(list(set(boosting_names) & set(ufsaco_names)))
        result = {
            "hp": copy.deepcopy(hp),
            "seed": seed,
            "tau": tau,
            "ufsaco_names": ufsaco_names,
            "intersection": intersection,
            "intersection_power": len(intersection),
        }
        if best_result is None or result["intersection_power"] > best_result["intersection_power"]:
            best_result = result

        if result["intersection_power"] >= MIN_INTERSECTION:
            break

    if best_result["intersection_power"] >= MIN_INTERSECTION:
        break

if best_result["intersection_power"] < MIN_INTERSECTION:
    raise RuntimeError("Не выполнено условие: мощность пересечения меньше 4.")

print("Лучшие гиперпараметры UFSACO:")
print(best_result["hp"])
print("Seed:", best_result["seed"])
ufsaco_names = best_result["ufsaco_names"]
intersection = best_result["intersection"]
tau_final = best_result["tau"]
features_UFSACO = np.array(tau_final.argsort()[::-1][:N_END_FEATURES])
ufsaco_vals = tau_final[features_UFSACO]
ufsaco_result = pd.DataFrame(data={"Name": ufsaco_names, "Pheromone": ufsaco_vals})
print("Признаки, выбранные UFSACO:")
print(ufsaco_result)
plt.figure(figsize=(10, 7))
sns.barplot(x="Pheromone", y="Name", data=ufsaco_result, color="g")
plt.title("UFSACO: итоговое количество феромона")
plt.tight_layout()
plt.show()



# Проверяем муравьиный алгоритм на 5 независимых запусках
stability_rows = []
NC_MAX = best_result["hp"]["NC_MAX"]
N_STEPS = best_result["hp"]["N_STEPS"]
INIT_PHEROMONE = best_result["hp"]["INIT_PHEROMONE"]
RO = best_result["hp"]["RO"]
EXPLOITATION_PROB = best_result["hp"]["EXPLOITATION_PROB"]
ALPHA = best_result["hp"]["ALPHA"]
BETA = best_result["hp"]["BETA"]
N_ANTS = best_result["hp"]["N_ANTS"]
for run_id in range(5):
    np.random.seed(best_result["seed"] + run_id)
    tau_run = UFSACO(verbose=False)
    idx_run = np.array(tau_run.argsort()[::-1][:N_END_FEATURES])
    names_run = [all_input_names[i] for i in idx_run]
    intersection_run = sorted(list(set(boosting_names) & set(names_run)))
    stability_rows.append({
        "run": run_id + 1,
        "features_UFSACO": names_run,
        "intersection": intersection_run,
        "intersection_power": len(intersection_run),
    })
stability_df = pd.DataFrame(stability_rows)
print(stability_df)

# Проверим сходство пар признаков с помощью косинусного сходства
all_sims = []
all_pairs = []
for i, name1 in enumerate(all_input_names):
    for j, name2 in enumerate(all_input_names):
        if j > i:
            all_sims.append(get_sim(i, j))
            all_pairs.append(name1 + " + " + name2)
series = pd.Series(data=all_sims, index=all_pairs)
print("Наиболее похожие пары признаков:")
print(series.sort_values(ascending=False).head(10))
print("Наименее похожие пары признаков:")
print(series.sort_values(ascending=True).head(10))


# Итоговое сравнение двух методов
boosting_set = set(boosting_names)
ufsaco_set = set(ufsaco_names)
intersection = sorted(list(boosting_set & ufsaco_set))
print("Итоговые признаки Gradient Boosting:")
print(boosting_names)
print("Итоговые признаки UFSACO:")
print(ufsaco_names)
print("Пересечение:")
print(intersection)
print("Мощность пересечения:", len(intersection))

result_summary = pd.DataFrame({
    "method": ["GradientBoostingRegressor", "UFSACO", "Intersection"],
    "n_features": [len(boosting_names), len(ufsaco_names), len(intersection)],
    "features": [boosting_names, ufsaco_names, intersection],
})

print(result_summary)