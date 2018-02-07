from __future__ import print_function

from forest import *
from nbmpi import GaussianNB

__all__ = ["get_model"
          ,"model_names"
          ]

_models = {}
def _register_model(cli_name, cls):
    if cli_name in _models:
        print('attempted to register duplicate model "{}"'.format(cli_name), file=sys.stderr)
        sys.exit(1)
    _models[cli_name] = cls

_register_model('nb', GaussianNB)
_register_model('rf', RandomForestRegressor)
_register_model('mf', MondrianForestRegressor)

def model_names():
    return _models.keys()

def get_model(cli_name):
    if cli_name in _models:
        return _models[cli_name]()
    else:
        return None

def get_model_id(cli_name):
    return sorted(_models.keys()).index(cli_name)

def get_cli_name(id_or_instance):
    for k, v in _models.iteritems():
        if id_or_instance.__class__ == v:
            return k
    return sorted(_models.keys())[int(id_or_instance)]
