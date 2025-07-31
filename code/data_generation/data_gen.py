import pandas as pd
import random

df = pd.read_csv('coffee.csv')
df.drop(columns=['Location.Country', 'Location.Altitude.Min', 'Location.Altitude.Max', 'Data.Owner', 'Data.Production.Number of bags', 'Data.Scores.Total'], inplace = True)
regions = list(set(df['Location.Region']))
species = list(set(df['Data.Type.Species']))
varieties = list(set(df['Data.Type.Variety']))
processings = list(set(df['Data.Type.Processing method']))
colors = ['Green', 'Blue-Green', 'Bluish-Green', 'None', 'Unknown', 'Rainbow', 'Red', 'Orange-Blue']
rows = []
for i in range(99010):
    region = random.choice(regions)
    altitude = random.randint(0, 200000)
    year = random.randint(2010, 2025)
    spec = random.choice(species)
    variety = random.choice(varieties)
    processing = random.choice(processings)
    weight = round(random.uniform(0, 20000), 4)
    aroma = round(random.uniform(0,10), 2)
    flavor = round(random.uniform(0, 10), 2)
    aftertaste = round(random.uniform(0, 10), 2)
    acidity = round(random.uniform(0, 10), 2)
    body = round(random.uniform(0, 10), 2)
    balance = round(random.uniform(0, 10), 2)
    uniformity = round(random.uniform(0, 10), 2)
    sweetness = round(random.uniform(0, 10), 2)
    moisture = round(random.uniform(0, 0.30), 2)
    color = random.choice(colors)
    rows.append({
        "Location.Region" : region,
        "Location.Altitude.Average" : altitude,
        "Year" : year,
        "Data.Type.Species" : spec,
        "Data.Type.Variety" : variety,
        "Data.Type.Processing method" : processing,
        "Data.Production.Bag weight" : weight,
        "Data.Scores.Aroma" : aroma,
        "Data.Scores.Flavor" : flavor,
        "Data.Scores.Aftertaste" : aftertaste,
        "Data.Scores.Acidity" : acidity,
        "Data.Scores.Body" : body,
        "Data.Scores.Balance" : balance,
        "Data.Scores.Uniformity" : uniformity,
        "Data.Scores.Sweetness" : sweetness,
        "Data.Scores.Moisture" : moisture,
        "Data.Color" : color
    })

new_df = pd.DataFrame(rows)
final_df = pd.concat([df, new_df], ignore_index = True)
final_df.to_csv("generated_coffee.csv", index=False, na_rep="nan")
