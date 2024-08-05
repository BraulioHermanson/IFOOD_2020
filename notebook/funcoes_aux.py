import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def inspect_outliers(dataframe, column, whisker_width=1.5):
    q1 = dataframe[column].quantile(0.25)
    # mediam = dataframe[column].quantile(0.5)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    return dataframe [
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

def pairplot(dataframe, columns, hue_column=None,alpha=0.5, corner=True,palette="tab10"):
    analysis = columns.copy() + [hue_column]

    sns.pairplot(
        dataframe[analysis],
        diag_kind="kde",
        hue=hue_column,
        plot_kws=dict(alpha=alpha),
        corner=corner,
        palette=palette
        );

def graficos_elbow_silhouette(X,random_state=42, intervalo_k=(2,11)):

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,5), tight_layout=True)

    elbow = {}
    silhouette = []

    k_range = range(*intervalo_k) # *=unpacking


    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(X)
        elbow[i] = kmeans.inertia_

        labels = kmeans.labels_
        silhouette.append(silhouette_score(X,labels))

    sns.lineplot(x=list(elbow.keys()), y=list(elbow.values()), ax=axs[0])
    axs[0].set_xlabel("K")
    axs[0].set_xlabel("Inertia")
    axs[0].set_title("Elbow Method")

    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].set_xlabel("K")
    axs[1].set_xlabel("Silhouette Score")
    axs[1].set_title("Silhouette Method")

    plt.show()


def visualizar_clusters(
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids=True, 
    mostrar_pontos=False,
    coluna_clusters=None,
):

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection="3d")
    
    cores = plt.cm.tab10.colors[:quantidade_cores]
    cores = ListedColormap(cores)

    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]
    
    ligar_centroids = mostrar_centroids
    ligar_pontos = mostrar_pontos
    
    for i, centroid in enumerate(centroids):
        if ligar_centroids: 
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(*centroid, f"{i}", fontsize=20, horizontalalignment="center", verticalalignment="center")
    
        if ligar_pontos:
            s = ax.scatter(x, y, z, c=coluna_clusters, cmap=cores)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    ax.set_title("Clusters")
    
    plt.show()


def plot_columns_percent_by_cluster(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster",
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor por cluster.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tupla com o número de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e a altura da figura, por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna com os números dos clusters, por padrão "cluster"
    """

    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(
            x=column_cluster,
            hue=col,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
        )

        n_clusters = dataframe[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=11,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()


def plot_columns_percent_hue_cluster(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster",
    palette="tab10",
):
    """Função para gerar gráficos de barras com a porcentagem de cada valor com cluster
    como hue.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    rows_cols : tuple, opcional
        Tupla com o número de linhas e colunas do grid de eixos, por padrão (2, 3)
    figsize : tuple, opcional
        Tupla com a largura e a altura da figura, por padrão (15, 8)
    column_cluster : str, opcional
        Nome da coluna com os números dos clusters, por padrão "cluster"
    palette : str, opcional
        Paleta a ser utilizada, por padrão "tab10"
    """
    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    if not isinstance(axs, np.ndarray):
        axs = np.array(axs)

    for ax, col in zip(axs.flatten(), columns):
        h = sns.histplot(
            x=col,
            hue=column_cluster,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
            palette=palette,
        )

        if dataframe[col].dtype != "object":
            h.set_xticks(range(dataframe[col].nunique()))

        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=11,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

        legend = h.get_legend()
        legend.remove()

    labels = [text.get_text() for text in legend.get_texts()]

    fig.legend(
        handles=legend.legend_handles,
        labels=labels,
        loc="upper center",
        ncols=dataframe[column_cluster].nunique(),
        title="Clusters",
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.show()