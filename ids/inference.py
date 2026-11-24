"""Single inference path shared by evaluation, CSV prediction and Streamlit."""
import json
from pathlib import Path
from .preprocess import ModelPreprocessor, Preprocessor
from .selection import blend_scores


class PredictionBundle:
    def __init__(self, processor, models, decisions=None, legacy=False):
        self.processor, self.models = processor, models
        self.decisions, self.legacy = decisions or {}, legacy
        self.names = ['xgboost', 'cnn'] if legacy else ['xgboost', 'cnn', 'ensemble']

    def threshold(self, name):
        return float(self.decisions.get('thresholds', {}).get(name, 0.5))

    def predict(self, raw, name):
        from .models import score_model
        if name not in self.names:
            raise ValueError(f'Unavailable model: {name}')
        def base(key):
            x = self.processor.transform(raw) if self.legacy else self.processor.transform(raw, key)
            return score_model(self.models[key], key, x)
        if name != 'ensemble':
            return base(name)
        return blend_scores(base('xgboost'), base('cnn'), self.decisions['ensemble'])

    @classmethod
    def load(cls, run):
        from .models import load_model
        run = Path(run)
        manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'trained':
            raise ValueError('Use a completed training run.')
        legacy = manifest.get('schema_version', 1) == 1
        processor_type = Preprocessor if legacy else ModelPreprocessor
        processor = processor_type.load(run / 'preprocess.npz')
        decisions = {} if legacy else json.loads((run / 'decisions.json').read_text(encoding='utf-8'))
        models = {name: load_model(run, name, manifest['config']['threads']) for name in ('xgboost', 'cnn')}
        return cls(processor, models, decisions, legacy)
