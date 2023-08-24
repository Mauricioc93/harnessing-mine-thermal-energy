import pandas as pd
import pyomo.core as pyomo
from math import *

def annuity(capex, n=30, wacc=0.03, u=None, cost_decrease=0):
    r"""Calculates the annuity of an initial investment 'capex', considering
    the cost of capital 'wacc' during a project horizon 'n'

    In case of a single initial investment, the employed formula reads:

    .. math::
        \text{annuity} = \text{capex} \cdot
            \frac{(\text{wacc} \cdot (1+\text{wacc})^n)}
            {((1 + \text{wacc})^n - 1)}

    In case of repeated investments (due to replacements) at fixed intervals
    'u', the formula yields:

    .. math::
        \text{annuity} = \text{capex} \cdot
                  \frac{(\text{wacc} \cdot (1+\text{wacc})^n)}
                  {((1 + \text{wacc})^n - 1)} \cdot \left(
                  \frac{1 - \left( \frac{(1-\text{cost\_decrease})}
                  {(1+\text{wacc})} \right)^n}
                  {1 - \left(\frac{(1-\text{cost\_decrease})}{(1+\text{wacc})}
                  \right)^u} \right)

    Parameters
    ----------
    capex : float
        Capital expenditure for first investment. Net Present Value (NPV) or
        Net Present Cost (NPC) of investment
    n : int
        Horizon of the analysis, or number of years the annuity wants to be
        obtained for (n>=1)
    wacc : float
        Weighted average cost of capital (0<wacc<1)
    u : int
        Lifetime of the investigated investment. Might be smaller than the
        analysis horizon, 'n', meaning it will have to be replaced.
        Takes value 'n' if not specified otherwise (u>=1)
    cost_decrease : float
        Annual rate of cost decrease (due to, e.g., price experience curve).
        This only influences the result for investments corresponding to
        replacements, whenever u<n.
        Takes value 0, if not specified otherwise (0<cost_decrease<1)
    Returns
    -------
    float
        annuity
    """
    if u is None:
        u = n

    if ((n < 1) or (wacc < 0 or wacc > 1) or (u < 1) or
            (cost_decrease < 0 or cost_decrease > 1)):
        raise ValueError("Input arguments for 'annuity' out of bounds!")

    return (
        capex * (wacc*(1+wacc)**n) / ((1 + wacc)**n - 1) *
        ((1 - ((1-cost_decrease)/(1+wacc))**n) /
         (1 - ((1-cost_decrease)/(1+wacc))**u)))

from scipy import optimize
import numpy as np

def slope_intercept(x1,y1,x2,y2):
    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1     
    return a,b


def segments_fit(X, Y, count):
    xmin = X.min()
    xmax = X.max()

    seg = np.full(count - 1, (xmax - xmin) / count)

    px_init = np.r_[np.r_[xmin, seg].cumsum(), xmax]
    py_init = np.array([Y[np.abs(X - x) < (xmax - xmin) * 0.01].mean() for x in px_init])

    def func(p):
        seg = p[:count - 1]
        py = p[count - 1:]
        px = np.r_[np.r_[xmin, seg].cumsum(), xmax]
        return px, py

    def err(p):
        px, py = func(p)
        Y2 = np.interp(X, px, py)
        return np.mean((Y - Y2)**2)

    r = optimize.minimize(err, x0=np.r_[seg, py_init], method='Nelder-Mead')
    return func(r.x)


# fric = 0.019
# rho = 997
# D_i = 0.107
# length = 100
# n_pump = 0.72
# # power_L = lambda m: (8*fric/(D_i**5*rho*np.pi**2)*(m)**2/1000)*m/rho/n_pump*1000
# power_L = lambda m: (8*fric/(D_i**5*rho**2*np.pi**2)*(m)**3/1000)/n_pump*1000

# X = np.linspace(0,12, 100)
# Y = power_L(X)
# nsec = 5
# px, py = segments_fit(X, Y, nsec)
# a_list = []
# b_list = []
# y_plot = np.zeros((len(X),nsec))

# for i, (x,y) in enumerate(zip(px,py)):
#     if i == 0:
#         continue
#     else:
#         a,b = slope_intercept(x,y,px[i-1],py[i-1])
#         a_list.append(a)
#         b_list.append(b)
#         y_plot[:,i-1] = X*a+b

# import pylab as pl
# pl.plot(X,y_plot)
# pl.plot(X, Y, ".")
# pl.plot(px, py, "-or")
# pl.show()

def get_entity(instance, name):
    """ Retrieve values (or duals) for an entity in a model instance.
    Args:
        instance: a Pyomo ConcreteModel instance
        name: name of a Set, Param, Var, Constraint or Objective
    Returns:
        a Pandas Series with domain as index and values (or 1's, for sets) of
        entity name. For constraints, it retrieves the dual values
    """
    # magic: short-circuit if problem contains a result cache
    if hasattr(instance, '_result') and name in instance._result:
        return instance._result[name].copy(deep=True)

    # retrieve entity, its type and its onset names
    try:
        entity = instance.__getattribute__(name)
        labels = _get_onset_names(entity)
    except AttributeError:
        return pd.Series(name=name)

    # extract values
    if isinstance(entity, pyomo.Set):
        if entity.dimen > 1:
            results = pd.DataFrame([v + (1,) for v in entity.value])
        else:
            # Pyomo sets don't have values, only elements
            results = pd.DataFrame([(v, 1) for v in entity.value])

        # for unconstrained sets, the column label is identical to their index
        # hence, make index equal to entity name and append underscore to name
        # (=the later column title) to preserve identical index names for both
        # unconstrained supersets
        if not labels:
            labels = [name]
            name = name + '_'

    elif isinstance(entity, pyomo.Param):
        if entity.dim() > 1:
            results = pd.DataFrame(
                [v[0] + (v[1],) for v in entity.iteritems()])
        elif entity.dim() == 1:
            results = pd.DataFrame(
                [(v[0], v[1]) for v in entity.iteritems()])
        else:
            results = pd.DataFrame(
                [(v[0], v[1].value) for v in entity.iteritems()])
            labels = ['None']

    elif isinstance(entity, pyomo.Expression):
        if entity.dim() > 1:
            results = pd.DataFrame(
                [v[0]+(v[1](),) for v in entity.iteritems()])
        elif entity.dim() == 1:
            results = pd.DataFrame(
                [(v[0], v[1]()) for v in entity.iteritems()])
        else:
            results = pd.DataFrame(
                [(v[0], v[1]()) for v in entity.iteritems()])
            labels = ['None']

    elif isinstance(entity, pyomo.Constraint):
        if entity.dim() > 1:
            # check whether all entries of the constraint have
            # an existing dual variable
            # in that case add to results
            results = pd.DataFrame(
                [key + (instance.dual[entity.__getitem__(key)],)
                 for (id, key) in entity.id_index_map().items()
                 if id in instance.dual._dict.keys()])
        elif entity.dim() == 1:
            results = pd.DataFrame(
                [(v[0], instance.dual[v[1]]) for v in entity.iteritems()])
        else:
            results = pd.DataFrame(
                [(v[0], instance.dual[v[1]]) for v in entity.iteritems()])
            labels = ['None']

    else:
        # create DataFrame
        if entity.dim() > 1:
            # concatenate index tuples with value if entity has
            # multidimensional indices v[0]
            results = pd.DataFrame(
                [v[0] + (v[1].value,) for v in entity.iteritems()])
        elif entity.dim() == 1:
            # otherwise, create tuple from scalar index v[0]
            results = pd.DataFrame(
                [(v[0], v[1].value) for v in entity.iteritems()])
        else:
            # assert(entity.dim() == 0)
            results = pd.DataFrame(
                [(v[0], v[1].value) for v in entity.iteritems()])
            labels = ['None']

    # check for duplicate onset names and append one to several "_" to make
    # them unique, e.g. ['sit', 'sit', 'com'] becomes ['sit', 'sit_', 'com']
    for k, label in enumerate(labels):
        if label in labels[:k] or label == name:
            labels[k] = labels[k] + "_"

    if not results.empty:
        # name columns according to labels + entity name
        results.columns = labels + [name]
        results.set_index(labels, inplace=True)

        # convert to Series
        results = results[name]
    else:
        # return empty Series
        results = pd.Series(name=name)
    return results


def get_entities(instance, names):
    """ Return one DataFrame with entities in columns and a common index.
    Works only on entities that share a common domain (set or set_tuple), which
    is used as index of the returned DataFrame.
    Args:
        instance: a Pyomo ConcreteModel instance
        names: list of entity names (as returned by list_entities)
    Returns:
        a Pandas DataFrame with entities as columns and domains as index
    """

    df = pd.DataFrame()
    for name in names:
        other = get_entity(instance, name)

        if df.empty:
            df = other.to_frame()
        else:
            index_names_before = df.index.names

            df = df.join(other, how='outer')

            if index_names_before != df.index.names:
                df.index.names = index_names_before

    return df


def list_entities(instance, entity_type):
    """ Return list of sets, params, variables, constraints or objectives
    Args:
        instance: a Pyomo ConcreteModel object
        entity_type: "set", "par", "var", "con" or "obj"
    Returns:
        DataFrame of entities
    Example:
        >>> data = read_excel('mimo-example.xlsx')
        >>> model = create_model(data, range(1,25))
        >>> list_entities(model, 'obj')  #doctest: +NORMALIZE_WHITESPACE
                                         Description Domain
        Name
        obj   minimize(cost = sum of all cost types)     []
    """

    # helper function to discern entities by type
    def filter_by_type(entity, entity_type):
        if entity_type == 'set':
            return isinstance(entity, pyomo.Set) and not entity.virtual
        elif entity_type == 'par':
            return isinstance(entity, pyomo.Param)
        elif entity_type == 'var':
            return isinstance(entity, pyomo.Var)
        elif entity_type == 'con':
            return isinstance(entity, pyomo.Constraint)
        elif entity_type == 'obj':
            return isinstance(entity, pyomo.Objective)
        elif entity_type == 'exp':
            return isinstance(entity, pyomo.Expression)
        else:
            raise ValueError("Unknown entity_type '{}'".format(entity_type))

    # create entity iterator, using a python 2 and 3 compatible idiom:
    # http://python3porting.com/differences.html#index-6
    try:
        iter_entities = instance.__dict__.iteritems()  # Python 2 compat
    except AttributeError:
        iter_entities = instance.__dict__.items()  # Python way

    # now iterate over all entities and keep only those whose type matches
    entities = sorted(
        (name, entity.doc, _get_onset_names(entity))
        for (name, entity) in iter_entities
        if filter_by_type(entity, entity_type))

    # if something was found, wrap tuples in DataFrame, otherwise return empty
    if entities:
        entities = pd.DataFrame(entities,
                                columns=['Name', 'Description', 'Domain'])
        entities.set_index('Name', inplace=True)
    else:
        entities = pd.DataFrame()
    return entities


def _get_onset_names(entity):
    """ Return a list of domain set names for a given model entity
    Args:
        entity: a member entity (i.e. a Set, Param, Var, Objective, Constraint)
                of a Pyomo ConcreteModel object
    Returns:
        list of domain set names for that entity
    Example:
        >>> data = read_excel('mimo-example.xlsx')
        >>> model = create_model(data, range(1,25))
        >>> _get_onset_names(model.e_co_stock)
        ['t', 'sit', 'com', 'com_type']
    """
    # get column titles for entities from domain set names
    labels = []

    if isinstance(entity, pyomo.Set):
        if entity.dimen > 1:
            # N-dimensional set tuples, possibly with nested set tuples within
            if entity.domain:
                # retreive list of domain sets, which itself could be nested
                domains = entity.domain.set_tuple
            else:
                try:
                    # if no domain attribute exists, some
                    domains = entity.set_tuple
                except AttributeError:
                    # if that fails, too, a constructed (union, difference,
                    # intersection, ...) set exists. In that case, the
                    # attribute _setA holds the domain for the base set
                    try:
                        domains = entity._setA.domain.set_tuple
                    except AttributeError:
                        # if that fails, too, a constructed (union, difference,
                        # intersection, ...) set exists. In that case, the
                        # attribute _setB holds the domain for the base set
                        domains = entity._setB.domain.set_tuple

            for domain_set in domains:
                labels.extend(_get_onset_names(domain_set))

        elif entity.dimen == 1:
            if entity.domain:
                # 1D subset; add domain name
                labels.append(entity.domain.name)
            else:
                # unrestricted set; add entity name
                labels.append(entity.name)
        else:
            # no domain, so no labels needed
            pass

    elif isinstance(entity, (pyomo.Param, pyomo.Var, pyomo.Expression,
                    pyomo.Constraint, pyomo.Objective)):
        if entity.dim() > 0 and entity._index:
            labels = _get_onset_names(entity._index)
        else:
            # zero dimensions, so no onset labels
            pass

    else:
        raise ValueError("Unknown entity type!")

    return labels