import matplotlib.pyplot as plt

__all__ = [ "nominal_value"
          , "std_dev"
          , "has_error_bars"
          , "plot"
          , "plot_continuous"
          , "legend"
          , "xlabel"
          , "ylabel"
          , "title"
          , "suptitle"
          , "grid"
          , "savefig"
          , "clf"
          , "xlim"
          , "ylim"
          ]

def _is_iterable(x):
    try:
        _ = iter(x)
        return True
    except TypeError:
        return False

def _map_getattr(items, attr, default):
    if _is_iterable(items):
        return map(lambda i: _map_getattr(i, attr, default), items)
    else:
        return getattr(items, attr, default(items))

def nominal_value(x):
    return _map_getattr(x, 'nominal_value', lambda i: i)

def std_dev(x):
    return _map_getattr(x, 'std_dev', lambda _: 0)

def has_error_bars(X):
    return hasattr(X[0], 'std_dev')

def plot(X, Y, line_format='-', **kwargs):
    if has_error_bars(X) or has_error_bars(Y):
        if has_error_bars(X):
            kwargs.update(xerr=std_dev(X))
        if has_error_bars(Y):
            kwargs.update(yerr=std_dev(Y))
        plt.errorbar(nominal_value(X), nominal_value(Y), fmt=line_format, **kwargs)
    else:
        plt.plot(nominal_value(X), nominal_value(Y), line_format, **kwargs)

def plot_continuous(f, min, max, **kwargs):
    X = np.linspace(min, max, 1000)
    Y = map(f, X)

    if has_error_bars(X) or has_error_bars(Y):
        kwargs.update(zorder=2, alpha=0.2)
    elif 'zorder' in kwargs and kwargs['zorder'] <= 2:
        kwargs['zorder'] = 3

    plot(X, Y, **kwargs)

legend = plt.legend
xlabel = plt.xlabel
ylabel = plt.ylabel
title = plt.title
suptitle = plt.suptitle
grid = plt.grid
savefig = plt.savefig
clf = plt.clf
xlim = plt.xlim
ylim = plt.ylim
