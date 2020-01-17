import pandas as pd
import matplotlib.pyplot as plt


def input_data_explore():
    input_df = pd.read_csv('../output/20newsGroup18828.csv')

    print(input_df.shape)

    fig, ax = plt.subplots(figsize=(17, 7))
    agg_df = input_df.groupby('category').count().sort_values('category')
    print(agg_df)
    fig_plot = agg_df.plot(kind='bar', ax=ax, title='The count of each category of the dataset',
                legend=False)
    fig_plot.set_xlabel('Categories')
    fig_plot.set_ylabel('Count')

    plt.tight_layout()
    # plt.autoscale()
    # plt.show()
    plt.savefig('../output/count.png', format='png')


def tf_dimension_naive():
    tf_data = {
        'KNN': [0.25, 0.26, 0.27, 0.30, 0.32, 0.3, 0.26],
        'SVM': [0.81, 0.76, 0.74, 0.68, 0.64, 0.42, 0.31],
        'NN': [0.85, 0.83, 0.78, 0.73, 0.65, 0.41, 0.34]
            }

    df = pd.DataFrame(tf_data, index=[6026, 4058, 2020, 1030, 508, 108, 54])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_plot = df.plot(kind='line', ax=ax, title='Term Frequency with Naive Dimension Reduction',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfnaive.png", format='png')
    plt.show()


def tf_dimension_svd():
    tf_data = {
        'KNN': [0.29, 0.31, 0.39, 0.45, 0.48, 0.55, 0.53],
        'SVM': [0.81, 0.79, 0.79, 0.78, 0.77, 0.65, 0.57],
        'NN': [0.81, 0.8, 0.79, 0.79, 0.77, 0.69, 0.64]
            }

    df = pd.DataFrame(tf_data, index=[6026, 4058, 2020, 1030, 508, 108, 54])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_plot = df.plot(kind='line', ax=ax, title='Term Frequency with SVD)',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfsvd.png", format='png')
    plt.show()


def tfidf_naive():
    tfidf_data = {
        'KNN': [0.76, 0.74, 0.69, 0.59, 0.49, 0.08, 0.06],
        'SVM': [0.87, 0.85, 0.82, 0.75, 0.68, 0.41, 0.32],
        'NN': [0.87, 0.85, 0.81, 0.74, 0.65, 0.45, 0.35]
    }

    df = pd.DataFrame(tfidf_data, index=[6026, 4058, 2020, 1030, 508, 108, 54])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_plot = df.plot(kind='line', ax=ax, title='TF-IDF with Naive Dimension Reduction',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfidfnaive.png", format='png')
    plt.show()


def tfidf_svd():
    tfidf_data = {
        'KNN': [0.77, 0.77, 0.58, 0.56, 0.54, 0.69, 0.7],
        'SVM': [0.87, 0.87, 0.86, 0.85, 0.83, 0.77, 0.74],
        'NN': [0.86, 0.86, 0.84, 0.82, 0.81, 0.79, 0.78]
            }

    df = pd.DataFrame(tfidf_data, index=[6026, 4058, 2020, 1030, 508, 108, 54])
    print(df)
    line_styles = [":", "-", "--"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig_plot = df.plot(kind='line', ax=ax, title='TF-IDF with SVD',
                       legend=True, style=line_styles)
    fig_plot.set_xlabel('Dimension')
    fig_plot.set_ylabel('Accuracy')
    plt.savefig("../output/tfidfsvd.png", format='png')
    plt.show()


def main():
    # input_data_explore()
    # tf_dimension_naive()
    # tf_dimension_svd()
    # tfidf_naive()
    tfidf_svd()


if __name__ == '__main__':
    main()
