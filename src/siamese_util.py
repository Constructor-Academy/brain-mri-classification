# #### Create samples for each class containing class samples and samples from other classes
#
# samples: for each class need to have num(same class, same class) = num(same class, other class)
#
def cls_sample_pairs(df, cls_column, cls_name, n_samples_per_class=10, column='image-name', random_state=47):
    """

    """
    pos_df = df[df[cls_column] == cls_name]
    neg_df = df[df[cls_column] != cls_name]
    replace = True

    # samples for same class image pairs
    cls_pos_df_first = pos_df.sample(n=n_samples_per_class, replace=replace, random_state=random_state)
    cls_pos_df_second = pos_df.sample(n=n_samples_per_class, replace=replace, random_state=2 * random_state)

    # samples for different classes image pairs
    cls_pos_df_third = pos_df.sample(n=n_samples_per_class, replace=replace, random_state=random_state)
    cls_neg_df = neg_df.sample(n=n_samples_per_class, replace=replace, random_state=3 * random_state)

    # create pairs and labels
    pairs = []
    labels = []

    for i in range(n_samples_per_class):
        same_pair = [
            cls_pos_df_first.iloc[i][column],
            cls_pos_df_second.iloc[i][column]
        ]
        other_pair = [cls_pos_df_third.iloc[i][column], cls_neg_df.iloc[i][column]]
        pairs += [same_pair, other_pair]
        labels += [1, 0]

    return pairs, labels


def create_pairs_all_classes(df, col, n_samples_per_class=10, column='image-name', random_state=47):
    """
    """
    cls = df[col].unique()
    all_pairs = []
    all_labels = []
    for c in cls:
        pairs, labels = cls_sample_pairs(df, col, c, n_samples_per_class, column, random_state)
        all_pairs += pairs
        all_labels += labels

    return all_pairs, all_labels


def download_embeddings(self, embedding_preds, label_names, vecs_filename, meta_filename):
    vecs_filename = f'{vecs_filename}.tsv'
    meta_filename = f'{meta_filename}.tsv'

    np.savetxt(vecs_filename, embedding_preds, delimiter='\t')

    out_m = open(meta_filename, 'w', encoding='utf-8')
    for name in label_names:
        out_m.write(f'{str(name)} \n')
        out_m.close()


def get_and_dowload_nembeddings(self, model, images, label_names, vecs_filename, meta_filename):
    embedding_preds = model.predict(images)
    self.download_embeddings(embedding_preds, label_names, vecs_filename, meta_filename)
    return embedding_preds
