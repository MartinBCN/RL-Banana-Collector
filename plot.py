from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px


p = Path('runs')


def load_df(file):
    with open(file, 'r') as data:
        data = json.load(data)
    df = pd.DataFrame({'Score': data['score'], 'Loss': data['loss'], 'ScoreMean': data['Mean Score'],
                       "Type": [file.stem] * len(data['loss']), 'Epoch': np.arange(len(data['loss']))})
    df['ScoreRolling'] = df['Score'].rolling(100).mean()
    return df


df = pd.concat([load_df(file) for file in p.glob('*.json')])

fig = px.line(df, x="Epoch", y="ScoreMean", color='Type')
fig.write_json('_includes/scores.json')
