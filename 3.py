import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(filepath, target_col, connected_cols, max_objects=5000):
    """
    Читаем датасет и выбираем только числовые признаки, исключая TARGET
    """
    raw = pd.read_csv(filepath)
    print("Размер исходного датасета:", raw.shape)

    numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
    features = [
        col for col in numeric_cols
        if col != target_col and col not in connected_cols
    ]

    df = raw[[target_col] + features].copy()
    df = df.dropna().reset_index(drop=True)
    print("Количество объектов после удаления пропусков:", len(df))
    print("Количество числовых признаков без target:", len(features))
    print("Размер данных для расчета:", df.shape)
    print(df.describe())

    # Ограничиваем выборку, тк иначе будет очень долго работать
    if max_objects is not None and len(df) > max_objects:
        df = df.sample(n=max_objects, random_state=42).reset_index(drop=True)
        print(f"Выборка ограничена до {max_objects} объектов.")

    return df, features


def scale_features(df, features, target_col):
    """
    Нормирование входных признаков методом MinMaxScaler
    """
    x_train = df[features].copy()
    y_train = df[target_col].copy()

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_train)
    x_train_scaled = x_scaler.transform(x_train)

    print("X shape:", x_train_scaled.shape)
    print("y shape:", y_train.shape)

    return x_train_scaled, y_train


def run_gradient_boosting(x_train, y_train, features, n_features=15):
    """
    Обучение модели GradientBoostingRegressor и отбор n_features наиболее важных признаков.
    """
    params = {
        "n_estimators": 300,        # количество деревьев
        "max_depth": 4,             # максимальная глубина каждого дерева
        "min_samples_split": 10,    # минимальное количество объектов, нужное для разбиения узла дерева
        "learning_rate": 0.01,      # скорость обучения
        "verbose": 0,               # (конфигурационный параметр) без подробного вывода процесса обучения
    }
    model_base_full = ensemble.GradientBoostingRegressor(**params)
    model_base_full.fit(x_train, y_train)

    base_feature_importance = copy.deepcopy(model_base_full.feature_importances_)
    base_feature_importance_sorted_idx = np.argsort(base_feature_importance)[::-1]
    base_feature_importance_sorted_idx = base_feature_importance_sorted_idx[0:n_features]

    boosting_names = [features[i] for i in base_feature_importance_sorted_idx]
    vals = base_feature_importance[base_feature_importance_sorted_idx]
    fimps = pd.DataFrame(data={"Name": boosting_names, "Vals": vals})

    return boosting_names, fimps


def plot_boosting_importance(fimps):
    """Визуализация важности признаков Gradient Boosting."""
    sns.set_context("paper", font_scale=1.3)
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Vals", y="Name", data=fimps, label="Total", color="b")
    plt.title("GradientBoostingRegressor: feature_importances_")
    plt.tight_layout()
    plt.show()


def get_sim(all_scaled, i, j, sim_cache, eps=1e-6):
    """
    Возвращает косинусное сходство между признаками i и j,
    используя кэширование в переданном словаре sim_cache.
    """
    pair = (min(i, j), max(i, j))
    if pair not in sim_cache:
        a = all_scaled[:, i]
        b = all_scaled[:, j]
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
        res = np.dot(np.asarray(a), np.asarray(b)) / denom
        sim_cache[pair] = np.abs(res)
    return sim_cache[pair]


def run_ufsaco(all_scaled, params, sim_cache, eps=1e-6, verbose=False):
    """
    Реализация метода муравьиной колонии для отбора признаков (UFSACO).
    """
    n_start_features = all_scaled.shape[1]
    nc_max = params["NC_MAX"]
    n_steps = params["N_STEPS"]
    init_pheromone = params["INIT_PHEROMONE"]
    ro = params["RO"]
    exploitation_prob = params["EXPLOITATION_PROB"]
    alpha = params["ALPHA"]
    n_ants = params["N_ANTS"]

    # Инициируем начальным значением феромона в каждом узле
    tau = init_pheromone * np.ones((n_start_features))

    # Внешний цикл - число эпох
    for count in range(nc_max):
        # Случайно размещаем муравьев по узлам с вероятностью, пропорциональной феромону
        tau_sum = sum(tau)
        p_dist = tau / tau_sum if tau_sum > 0 else np.ones_like(tau) / n_start_features
        ants_pos = np.random.choice(n_start_features, size=n_ants, p=p_dist)

        # Очищаем счетчик посещений узлов для каждого узла
        visits = np.zeros((n_start_features))

        # Очищаем множество посещенных узлов каждым k-ым агентом из каждого i-ого узла
        nodes_visited = {(k, i): set() for k in range(n_ants) for i in range(n_start_features)}

        # Внутренний цикл - длина пути, который проходят все муравьи
        for iter in range(n_steps):
            # k - номер текущего муравья
            for k in range(n_ants):
                # На каком i-ом узле находится k-ый муравей
                i = ants_pos[k]

                # Множество посещенных узлов
                visited = nodes_visited[(k, i)]

                # Множество непосещенных узлов
                unvisited = list((set(range(n_start_features)) - visited) - {i})

                if not unvisited:
                    continue

                # Вычисляем желательность перехода к непосещенным узлам
                node_score = np.array([
                    tau[j] / (np.power(get_sim(all_scaled, i, j, sim_cache, eps), alpha) + eps)
                    for j in unvisited
                ])

                # Какой шаг выполняем - exploration или exploitation?
                q = np.random.uniform()
                if q <= exploitation_prob:
                    # EXPLOITATION (жадный выбор самого привлекательного)
                    if verbose:
                        print("EXPLOITATION")
                    # Переходим в узел с максимальной желательностью
                    jj = np.argmax(node_score)
                else:
                    # EXPLORATION (стохастический выбор пропорционально вероятностям)
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

        # Обновление и испарение феромонов по окончании эпохи
        total_visits = sum(visits)
        if total_visits == 0:
            tau = (1 - ro) * tau
        else:
            tau = (1 - ro) * tau + (visits / total_visits)

    return tau


def find_best_ufsaco_params(boosting_names, all_scaled, all_input_names, param_grid, n_end_features, min_intersection,
                            eps=1e-6):
    """
    Перебор гиперпараметров и случайных инициализаций алгоритма UFSACO
    для поиска стабильного пересечения с результатами бустинга (не менее min_intersection).
    """
    best_result = None
    sim_cache = {}

    for hp in param_grid:
        for seed in range(100, 130):
            np.random.seed(seed)
            tau = run_ufsaco(all_scaled, hp, sim_cache, eps, verbose=False)
            features_UFSACO = np.array(tau.argsort()[::-1][:n_end_features])
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

            if result["intersection_power"] >= min_intersection:
                break

        if best_result["intersection_power"] >= min_intersection:
            break

    if best_result["intersection_power"] < min_intersection:
        raise RuntimeError(f"Не выполнено условие: мощность пересечения меньше {min_intersection}.")

    return best_result, sim_cache


def plot_ufsaco_pheromones(ufsaco_result):
    """Визуализация финального распределения феромонов у признаков."""
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Pheromone", y="Name", data=ufsaco_result, color="g")
    plt.title("UFSACO: итоговое количество феромона")
    plt.tight_layout()
    plt.show()


def run_stability_check(best_result, boosting_names, all_scaled, all_input_names, n_end_features, sim_cache, eps=1e-6):
    """
    Запуск UFSACO на 5 независимых инициализациях с найденными лучшими параметрами
    для оценки стабильности получаемого набора признаков.
    """
    stability_rows = []
    hp = best_result["hp"]
    base_seed = best_result["seed"]

    for run_id in range(5):
        np.random.seed(base_seed + run_id)
        tau_run = run_ufsaco(all_scaled, hp, sim_cache, eps, verbose=False)
        idx_run = np.array(tau_run.argsort()[::-1][:n_end_features])
        names_run = [all_input_names[i] for i in idx_run]
        intersection_run = sorted(list(set(boosting_names) & set(names_run)))
        stability_rows.append({
            "run": run_id + 1,
            "features_UFSACO": names_run,
            "intersection": intersection_run,
            "intersection_power": len(intersection_run),
        })
    return pd.DataFrame(stability_rows)


def analyze_pairwise_similarities(all_input_names, all_scaled, sim_cache, eps=1e-6):
    """
    Анализ и ранжирование пар признаков по степени их сходства (косинусной мере).
    """
    all_sims = []
    all_pairs = []
    for i, name1 in enumerate(all_input_names):
        for j, name2 in enumerate(all_input_names):
            if j > i:
                all_sims.append(get_sim(all_scaled, i, j, sim_cache, eps))
                all_pairs.append(name1 + " + " + name2)
    series = pd.Series(data=all_sims, index=all_pairs)
    return series


def main():
    FILEPATH = "Vote Ai.csv"
    TARGET = "Vote_Percentage"
    CONNECTED_COLS = ["Voter_Turnout"]
    MAX_OBJECTS = 5000
    N_FEATURES = 15
    MIN_INTERSECTION = 4
    EPS = 1e-6

    # 1. Загрузка и первичная обработка данных
    df, features = load_and_preprocess_data(FILEPATH, TARGET, CONNECTED_COLS, MAX_OBJECTS)

    # Проверка линейной корреляции признаков с таргетом
    raw_full = pd.read_csv(FILEPATH)
    corr = raw_full[features + [TARGET]].corr(numeric_only=True)[TARGET].sort_values(ascending=False)
    print("\nКорреляция признаков с целевой переменной:")
    print(corr.head(15))

    # 2. Нормирование признаков
    x_train, y_train = scale_features(df, features, TARGET)

    # 3. Отбор признаков через GradientBoostingRegressor
    boosting_names, fimps = run_gradient_boosting(x_train, y_train, features, N_FEATURES)
    print("\nПризнаки, выбранные GradientBoostingRegressor:")
    print(fimps)
    plot_boosting_importance(fimps)

    # 4. Поиск наилучших гиперпараметров для UFSACO
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

    print("\nПоиск гиперпараметров UFSACO...")
    best_result, sim_cache = find_best_ufsaco_params(
        boosting_names=boosting_names,
        all_scaled=x_train,
        all_input_names=features,
        param_grid=param_grid,
        n_end_features=N_FEATURES,
        min_intersection=MIN_INTERSECTION,
        eps=EPS
    )

    print("\nЛучшие гиперпараметры UFSACO:")
    print(best_result["hp"])
    print("Выбранный seed:", best_result["seed"])

    ufsaco_names = best_result["ufsaco_names"]
    tau_final = best_result["tau"]
    features_UFSACO_idx = np.array(tau_final.argsort()[::-1][:N_FEATURES])
    ufsaco_vals = tau_final[features_UFSACO_idx]
    ufsaco_result = pd.DataFrame(data={"Name": ufsaco_names, "Pheromone": ufsaco_vals})

    print("\nПризнаки, выбранные UFSACO:")
    print(ufsaco_result)
    plot_ufsaco_pheromones(ufsaco_result)

    # 5. Проверка стабильности алгоритма UFSACO на 5 прогонах
    stability_df = run_stability_check(
        best_result=best_result,
        boosting_names=boosting_names,
        all_scaled=x_train,
        all_input_names=features,
        n_end_features=N_FEATURES,
        sim_cache=sim_cache,
        eps=EPS
    )
    print("\nАнализ стабильности на 5 запусках:")
    print(stability_df)

    # 6. Анализ попарных сходств признаков
    similarities_series = analyze_pairwise_similarities(features, x_train, sim_cache, EPS)
    print("\nНаиболее похожие пары признаков:")
    print(similarities_series.sort_values(ascending=False).head(10))
    print("\nНаименее похожие пары признаков:")
    print(similarities_series.sort_values(ascending=True).head(10))

    # 7. Итоговое сравнение работы двух методов отбора
    boosting_set = set(boosting_names)
    ufsaco_set = set(ufsaco_names)
    final_intersection = sorted(list(boosting_set & ufsaco_set))

    print("\n================ Итоговое сравнение методов ================")
    print("Итоговые признаки Gradient Boosting:", boosting_names)
    print("Итоговые признаки UFSACO:", ufsaco_names)
    print("Пересечение:", final_intersection)
    print("Мощность пересечения:", len(final_intersection))

    result_summary = pd.DataFrame({
        "method": ["GradientBoostingRegressor", "UFSACO", "Intersection"],
        "n_features": [len(boosting_names), len(ufsaco_names), len(final_intersection)],
        "features": [boosting_names, ufsaco_names, final_intersection],
    })
    print("\nСводная таблица результатов:")
    print(result_summary)


if __name__ == "__main__":
    main()
