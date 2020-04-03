import click
import pandas as pd
from pathlib import Path

from lib.clustering import LdaClusterer, TfidfSvdKmeans 


@click.command()
@click.option('--file', '-f', prompt='Path to .csv',
              help='Path to .csv file from which to load data')
@click.option('--column', '-c', default='message_body',
              help='Name of column that contains message body')
@click.option('--cluster_type', '-c', default='lda',
              help='Choose lda or kmeans clustering')
@click.option('--n_clusters', '-n', default=5,
              help='Number of clusters')
def main(file, column, cluster_type, n_clusters):
    path_to_csv = Path(file)

    df = pd.read_csv(path_to_csv)
    texts = df[column].fillna('NULL').tolist()

    if cluster_type == 'lda':
        lda_clusterer = LdaClusterer(n_clusters)
        lda_clusterer.fit(texts)
        labels = lda_clusterer.predict(texts)

    elif cluster_type == 'kmeans':
        kmeans_clusterer = TfidfSvdKmeans(n_clusters)
        kmeans_clusterer.fit(texts)
        labels = kmeans_clusterer.predict(texts)
    else:
        print('Invalid cluster type. Choose lda or kmeans')
        return None

    df['cluster'] = labels
    df.to_csv('output.csv')


if __name__ == '__main__':
    main()
