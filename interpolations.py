import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

def interpolate_gp(patient_per_hour, column, **kwargs):
    # long_term_trend_kernel = RBF(
    #     length_scale=12.0,
    #     length_scale_bounds=(1, 1e2)
    # )
    # noise_kernel = WhiteKernel(
    #     noise_level=1e-2,
    #     noise_level_bounds=(1e-5, 1e1)
    # )
    # kernel = long_term_trend_kernel + noise_kernel

    kernel = ConstantKernel(
        constant_value=25.0,
        constant_value_bounds=(1e-2, 60)
    ) * RBF(
        length_scale=6.0,
        length_scale_bounds=(1, 24*7)
    )

    X = patient_per_hour[['unix_timestamp', column]].dropna()
    y = X[column]
    gaussian_process = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=0.5 ** 2, **kwargs)
    gaussian_process.fit(X[['unix_timestamp']], y)
    mean_y_pred, std_y_pred = gaussian_process.predict(patient_per_hour[['unix_timestamp']], return_std=True)
    return mean_y_pred, std_y_pred, gaussian_process

def extrapolate_cv(patient_per_hour, column, n_splits=5, is_split_per_column=False, model='gp', **kwargs):
    kernel = ConstantKernel(
        constant_value=25.0,
        constant_value_bounds=(1e-2, 60)
        # constant_value=7.4,
        # constant_value_bounds=(6.8, 7.8)
    ) * RBF(
        length_scale=6.0,
        length_scale_bounds=(1, 24*7)
    )

    model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=0.5 ** 2, **kwargs)
    mean_y_preds = []
    std_y_preds = []

    if is_split_per_column:
        X = patient_per_hour[['unix_timestamp', column]].dropna()
        y = X[column]
        
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in splitter.split(X):
            train, _ = X.values[train_index], X.values[test_index]
            X_train, y_train = train[:,:1], train[:,1]
            model.fit(X_train, y_train)
            mean_y_pred, std_y_pred = model.predict(
                patient_per_hour[['unix_timestamp']].values,
                return_std=True
            )
            mean_y_preds.append(mean_y_pred)
            std_y_preds.append(std_y_pred)
    else: # split by the same windows
        splitter = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in splitter.split(patient_per_hour):
            train = patient_per_hour.iloc[train_index][['unix_timestamp', column]].dropna()
            X_train, y_train = train[['unix_timestamp']].values, train[column].values
            model.fit(X_train, y_train)
            mean_y_pred, std_y_pred = model.predict(
                patient_per_hour[['unix_timestamp']].values,
                return_std=True
            )
            mean_y_preds.append(mean_y_pred)
            std_y_preds.append(std_y_pred)
    
    return np.array(mean_y_preds), np.array(std_y_preds)

def draw_extrapolation(
    patient_per_hour,
    columns,
    n_splits=5,
    is_split_per_column=True,
    show_interval=False,
    ax=None
):
    if ax is None:
        ax = plt.gca()

    x_support = patient_per_hour['pseudchartTime']
    for i, col in enumerate(columns):
        mean_y_pred, std_y_pred = extrapolate_cv(patient_per_hour, col, n_splits, is_split_per_column)
        ax.plot(x_support, mean_y_pred.T, alpha=.5, color=f"C{i}")
        ax.scatter(
            x='pseudchartTime',
            y=col,
            data=patient_per_hour,
            color=f"C{i}",
            label=col
        )
        if show_interval:
            for mu, sigma in zip(mean_y_pred, std_y_pred):
                ax.fill_between(
                    x_support,
                    mu - 2 * sigma,
                    mu + 2 * sigma,
                    alpha=.1, color=f"C{i}"
                )
    ax.legend()

def interpolate(patient_per_hour, column, **kwargs):
    X = patient_per_hour[['unix_timestamp', column]].dropna()
    model = make_pipeline(
        SplineTransformer(**kwargs),
        Ridge(alpha=1e-3)
    )
    model.fit(
        X[['unix_timestamp']],
        X[column]
    )
    y_pred = model.predict(patient_per_hour[['unix_timestamp']])
    return y_pred

def interpolate_cv(patient_per_hour, column, n_splits=5, shuffle=False, ts_split=False, **kwargs):
    X = patient_per_hour[['unix_timestamp', column]].dropna()
    model = make_pipeline(
        SplineTransformer(**kwargs),
        Ridge(alpha=1e-3)
    )
    y_preds = []
    if ts_split:
        splitter = TimeSeriesSplit(n_splits=n_splits)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=shuffle)
    for train_index, test_index in splitter.split(X):
        train, _ = X.values[train_index], X.values[test_index]
        X_train, y_train = train[:,0], train[:,1]
        model.fit(X_train.reshape(-1, 1), y_train)
        y_preds.append(model.predict(patient_per_hour[['unix_timestamp']].values))
    return np.array(y_preds)

def interpolate_cv_bagging(patient_per_hour, column, n_splits=5, **kwargs):
    X = patient_per_hour[['unix_timestamp', column]].dropna()
    model = make_pipeline(
        SplineTransformer(**kwargs),
        Ridge(alpha=1e-3)
    )
    y_preds = []
    bag = BaggingRegressor(model, n_estimators=n_splits, max_samples=2/3, bootstrap=False)
    bag.fit(X[['unix_timestamp']], X[column])
    for est in bag.estimators_:
        y_preds.append(est.predict(patient_per_hour[['unix_timestamp']]))
    return np.array(y_preds)

def predict_from_interpolations(patient_per_hour, features, target_column='ph', method='gp'):
    interpolated = patient_per_hour.copy()
    for col in features:
        if method == 'gp':
            y_interpolated, _, _ = interpolate_gp(patient_per_hour, col)
        elif method == 'cubic':
            y_interpolated = interpolate(patient_per_hour, col)
        else:
            raise NotImplementedError

        interpolated[col] = np.where(
            interpolated[col].isna().values,
            y_interpolated,
            interpolated[col]
        )

    X_train = interpolated.dropna(subset=target_column)
    reg = LinearRegression()
    reg.fit(X_train[features], X_train[target_column])
    y_pred = reg.predict(interpolated[features])
    return y_pred

def draw_interpolation(
    patient_per_hour,
    columns,
    cross_validate=True,
    n_splits=3,
    method='cubic',
    is_shuffled=False,
    is_summarised=True,
    is_ts_split=False,
    ax=None
):
    if ax is None:
        ax = plt.gca()

    x_support = patient_per_hour['pseudchartTime']
    for i, col in enumerate(columns):
        if method == 'gp':
            mean_y_pred, std_y_pred, mdl = interpolate_gp(patient_per_hour, col)
            y_pred = mean_y_pred

            ax.fill_between(
                x_support,
                mean_y_pred - 2 * std_y_pred,
                mean_y_pred + 2 * std_y_pred,
                alpha=.1, color=f"C{i}"
            )

            if not is_summarised:
                ax.plot(
                    x_support,
                    mdl.sample_y(patient_per_hour[['unix_timestamp']], n_samples=n_splits),
                    alpha=.3,
                    color=f"C{i}"
                )
        elif method == 'cubic':
            if cross_validate:
                y_preds = interpolate_cv(
                    patient_per_hour, col, n_splits=n_splits, shuffle=is_shuffled, ts_split=is_ts_split
                )
                mean_y_pred, std_y_pred = y_preds.mean(axis=0), y_preds.std(axis=0) / np.sqrt(n_splits)
                y_pred = mean_y_pred

                if is_summarised:
                    ax.fill_between(
                        x_support,
                        mean_y_pred - 2 * std_y_pred,
                        mean_y_pred + 2 * std_y_pred,
                        alpha=.1, color=f"C{i}"
                    )
                else:
                    ax.plot(x_support, y_preds.T, alpha=.5, color=f"C{i}")
            else:
                y_pred = interpolate(patient_per_hour, col)

        ax.plot(x_support, y_pred, lw=2, color=f"C{i}")
        ax.scatter(
            x='pseudchartTime',
            y=col,
            data=patient_per_hour,
            color=f"C{i}",
            label=col
        )
    ax.legend()