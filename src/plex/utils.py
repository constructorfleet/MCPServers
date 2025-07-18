import jsonpickle
from plexapi.server import PlexServer
from plexapi.client import PlexClient
from plexapi.base import PlexObject

def initialize_pickle(url: str):
    class PlexJsonHandler(jsonpickle.handlers.BaseHandler):
        def restore(self, obj):
            pass

        def flatten(self, obj, data):
            # Fix url raise
            if isinstance(obj, PlexClient):
                setattr(obj, "_baseurl", url)
            #remove methods, private fields etc
            members = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__") and not attr.startswith("_")]
            for normal_field in members:
                # we use context flatten - so its called handlers for given class
                data[normal_field] = self.context.flatten(getattr(obj, normal_field), {})
            return data
    jsonpickle.handlers.registry.register(PlexObject, PlexJsonHandler, True)

def to_json(obj):
   return jsonpickle.encode(obj, unpicklable=False, make_refs=False)

def from_json(json):
    return jsonpickle.decode(json)
