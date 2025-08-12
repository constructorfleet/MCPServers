from dataclasses import dataclass
from enum import StrEnum
import os
from typing import Any, Dict, Optional
import logging

from plexapi.exceptions import Unauthorized
from plexapi.server import PlexServer
from plexapi.library import MovieSection, ShowSection
from plexapi.video import Movie as PlexAPIMovie, Episode as PlexAPIEpisode
from plex.utils import as_dict, sort_by_similarity

logger = logging.getLogger(__name__)

