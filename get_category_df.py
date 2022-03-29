from sklearn.preprocessing import LabelEncoder

def get_category_df(train_set, testing_set):
    train_set = train_set[['item_condition_id', 'category_name', 'brand_name']]
    testing_set = testing_set[['item_condition_id', 'category_name', 'brand_name']]

    le = LabelEncoder()
    train_set['category_name'] = train_set[['category_name']].apply(le.fit_transform)
    train_set['brand_name'] = train_set[['brand_name']].apply(le.fit_transform)

    testing_set['category_name'] = testing_set[['category_name']].apply(le.fit_transform)
    testing_set['brand_name'] = testing_set[['brand_name']].apply(le.fit_transform)

    return (train_set, testing_set)