import zipfile, os
import re
import pandas as pd
from collections import Counter

def unzip_files(zip_files:list):
  for zip_path in zip_files:
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
      zip_file.extractall("/content/")
    os.remove(zip_path)

def cleaner(data, column_name:str) -> pd.DataFrame:
  q1 = data[column_name].quantile(q=0.01) 
  q2 = data[column_name].quantile(q=0.99)
  cleaned_data = data[(data[column_name] > q1) & (data[column_name] < q2)].reset_index(drop=True)
  return cleaned_data

def delete_cols(data, columns_to_drop):
  result_df = data.copy()
  for col in columns_to_drop:
    if col in result_df.columns:
     result_df.drop(col, axis=1, inplace=True)
  return result_df


def get_info_features(data):
  feature_data = data["features"]
  pattern = re.compile(r'[ \]\[\'\"]+')
  
  all_features = []
  unique_features = []
  data_with_features = data.copy()

  for idx, row in feature_data.items():
    row = str(row).lower()
    cleaned_features = pattern.sub('', row)
    idx_feats = set(cleaned_features.split(","))
    all_features.extend(idx_feats)

  unique_features = set(all_features)
  return all_features, unique_features

def features_encoder(data, top20:dict):
  creative_df = data.copy()

  for feature_name in top20.keys():
      creative_df[feature_name] = 0

  pattern = re.compile(r'[ \]\[\'\"]+')
  for idx, raw_features_list in creative_df['features'].items():
    features_as_string = str(raw_features_list).lower()
    cleaned_concatenated_features = pattern.sub('', features_as_string)
    individual_features = [f.strip() for f in cleaned_concatenated_features.split(',') if f.strip()]

    for feature_token in individual_features:
        if feature_token in top20.keys():
            creative_df.loc[idx, feature_token] = 1
  return creative_df 


def run_preprocessing(data:pd.DataFrame):
  """Входные данные - исходный датасет pd.DataFrame.
  Очистка от выбросов по цене, удаление ненужных колонок,
  Очистка от выбросов для ванн и спален, энкодинг interest_level;
  Парсинг признаков из колонки "features", 

  """
  columns_to_drop = ['display_address', 'street_address', 'building_id',
                     'created', 'listing_id','manager_id', 'photos','description']
  data_cleaned = delete_cols(data, columns_to_drop)
  data_cleaned = cleaner(data_cleaned, "price")

  # очистка от выбросов ванн и спален
  # значение 10 ванн - скорее всего опечатка (должно быть 1 за такую-то цену)
  data_cleaned.loc[data_cleaned["bathrooms"] == 10, "bathrooms"] = 1
  if (data_cleaned.loc[(data_cleaned["bathrooms"] == 10)]).empty:
    print("Ванны и спальни почищены от выбросов.")

  interest_mapping = {
    "low": 0,
    "medium": 1,
    "high": 2
  }
  data_cleaned["interest_level"] = data_cleaned["interest_level"].map(interest_mapping)

  all_features, unique_features = get_info_features(data_cleaned)

  cnt = Counter()
  for f in all_features:
    if f != '':
      cnt[f] +=1
  top20 = cnt.most_common(20)
  top20 = dict(top20)

  creative_df = features_encoder(data_cleaned, top20)
  assert creative_df["diningroom"].iloc[0] == 1, "Wrong"
  assert creative_df["pre-war"].iloc[0] == 1, "Wrong"
  assert creative_df["laundryinbuilding"].iloc[0] == 1, "Wrong"
  assert creative_df["dishwasher"].iloc[0] == 1, "Wrong"
  assert creative_df["hardwoodfloors"].iloc[0] == 1, "Wrong"
  assert creative_df["dogsallowed"].iloc[0] == 1, "Wrong"
  assert creative_df["catsallowed"].iloc[0] == 1, "Wrong"
  print("Всё assert-ы прошли - признаки из features энкодированы корректно.")

  creative_df = delete_cols(creative_df, ["features"])

  creative_df['bathrooms'] = creative_df['bathrooms'].astype('int')
  creative_df['bedrooms'] = creative_df['bedrooms'].astype('int')

  return creative_df


